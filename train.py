from transformers import WhisperConfig, WhisperModel, CLIPConfig, CLIPModel, AutoTokenizer, AutoModel, AutoForCausalLM
import whisper
import torch
import torch.nn as nn
from dataset_utils import _transform, create_dataloaders
import time
from tqdm.auto import tqdm
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn


class TextClassificationModel(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(TextClassificationModel, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.fc = nn.Linear(hidden_size, num_classes)  # Fully connected layer

    def forward(self, x):
        # x: [batch_size, sequence_length, hidden_size]
        x = x.permute(0, 2, 1)  # Change to [batch_size, hidden_size, sequence_length]
        x = self.global_avg_pool(x)  # [batch_size, hidden_size, 1]
        x = x.squeeze(-1)  # [batch_size, hidden_size]
        x = self.fc(x)  # [batch_size, num_classes]
        return x


class ImageAudioModel(nn.Module):
    def __init__(self, num_classes):
        super(ImageAudioModel, self).__init__()
        # self.whisper_config = WhisperConfig()
        # self.audio_encoder = WhisperModel(self.whisper_config).to(device)
        # self.image_config = CLIPConfig()
        # self.image_encoder = CLIPModel(self.image_config).to(device)
        # self.image_encoder = AutoModel.from_pretrained("nvidia/E-RADIO", trust_remote_code=True).to(device)
        self.image_encoder_name = "openai/clip-vit-base-patch32"
        self.audio_encoder_name = "openai/whisper-base"
        self.llm_name = "llmware/bling-sheared-llama-1.3b-0.1"
        self.image_encoder = CLIPModel.from_pretrained(self.image_encoder_name).to(device)
        self.audio_encoder = WhisperModel.from_pretrained(self.audio_encoder_name).to(device)

        self.image_h_dim = 512
        self.audio_h_dim = 256

        self.image_seq_len = 50
        self.audio_seq_len = 1500
        
        self.aligned_seq_len = 32
    
        llm = AutoForCausalLM.from_pretrained(self.llm_name)
        self.embed_tokens = llm.model.embed_tokens
        self.llm_dim = self.embed_tokens.shape[1]

        self.project_image = nn.Conv1d(self.image_seq_len, self.aligned_seq_len,
                                       kernel_size=1, stride=1).to(device)

        self.project_audio = nn.Conv1d(self.audio_seq_len, self.aligned_seq_len,
                                       kernel_size=1, stride=1).to(device)

        self.transform_image_to_hidden = nn.Linear(self.image_h_dim,
                                                   self.llm_dim).to(device)
        self.transform_audio_to_hidden = nn.Linear(self.audio_h_dim,
                                                   self.llm_dim).to(device)
        
        self.image_align_attention = nn.MultiheadAttention(256,
                                                           4 * 2,
                                                           dropout=0,
                                                           add_bias_kv=False,
                                                           add_zero_attn=False).to(device)
        self.audio_align_attention = nn.MultiheadAttention(256,
                                                           4 * 2,
                                                           dropout=0,
                                                           add_bias_kv=False,
                                                           add_zero_attn=False).to(device)

        self.num_classes = num_classes
        self.final_classifier = TextClassificationModel(256, num_classes).to(device)

    def forward(self, inputs, inputs_common):
        image_features = self.image_encoder.visual_projection(
            self.image_encoder.vision_model(inputs['images']).last_hidden_state)
        audio_features = self.audio_encoder.encoder(inputs['audios']).last_hidden_state

        text_embeddings = self.embed_tokens(inputs_common['input_ids'])

        token_embeddings = self.embed_tokens.weight.unsqueeze(0).repeat(
            text_embeddings.size(0), 1, 1).transpose(0, 1)

        audio_starts = self.embed_tokens(inputs_common['audio_starts'])
        audio_ends = self.embed_tokens(inputs_common['audio_ends'])

        # audio_features = self.project_audio(audio_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        audio_features = self.project_audio(audio_features)
        audio_features = self.transform_audio_to_hidden(audio_features)

        audio_features = self.audio_align_attention(audio_features.transpose(0, 1),
                                                    token_embeddings, token_embeddings)[0].transpose(0, 1).contiguous()

        audio_inputs = torch.cat([torch.cat([audio_starts, audio_features], dim=1), audio_ends], dim=1)
        text_embeddings = torch.cat([text_embeddings, audio_inputs], dim=1)

        image_starts = self.embed_tokens(inputs_common['image_starts'])
        image_ends = self.embed_tokens(inputs_common['image_ends'])

        # image_features = self.project_image(image_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        image_features = self.project_image(image_features)
        image_features = self.transform_image_to_hidden(image_features)
        image_features = self.image_align_attention(image_features.transpose(0, 1),
                                                    token_embeddings, token_embeddings)[0].transpose(0, 1).contiguous()

        image_inputs = torch.cat([torch.cat([image_starts, image_features], dim=1), image_ends], dim=1)

        text_embeddings = torch.cat(
            [torch.cat([text_embeddings, image_inputs], dim=1),
             self.embed_tokens(inputs_common['input_ide'])], dim=1)

        input_tensor = text_embeddings
        output = self.final_classifier(input_tensor)
        return output


def dummy_test():
    audio = whisper.load_audio("./aud.wav")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    all_audio_mels = mel.unsqueeze(0)  # (1, 80, 3000)

    preprocess = _transform(224)

    import matplotlib.pyplot as plt
    frame = preprocess(plt.imread("./img.jpg"))
    frame = frame.unsqueeze(0)

    bs = 1
    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
    inputs_common = {
        'image_starts': torch.tensor([tokenizer('<image>')['input_ids']] * bs, dtype=torch.int),
        'image_ends': torch.tensor([tokenizer('</image>')['input_ids']] * bs, dtype=torch.int),
        'audio_starts': torch.tensor([tokenizer('<audio>')['input_ids']] * bs, dtype=torch.int),
        'audio_ends': torch.tensor([tokenizer('</audio>')['input_ids']] * bs, dtype=torch.int),
        'input_ids': torch.tensor([tokenizer('<text>')['input_ids']] * bs, dtype=torch.int),
        'input_ide': torch.tensor([tokenizer('</text>')['input_ids']] * bs, dtype=torch.int)
    }
    inputs = {'audios': all_audio_mels, 'images': frame}

    inputs = {k: inputs[k].to(device) for k in inputs}
    inputs_common = {k: inputs_common[k].to(device) for k in inputs_common}

    model = ImageAudioModel(num_classes=30)
    model = model.to(device)
    output = model(inputs, inputs_common)
    print(output.shape)  # Should be [batch_size, num_classes]


def evaluate(model, dataloader, step):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader),
                            total=len(dataloader),
                            leave=False,
                            desc="Evaluating on validation data"):
            data = {k: data[k].to(device) for k in data}
            output = model(data, inputs_common)
            loss = criterion(output, data["materials"])
            total_loss += loss.item()
            total += data["materials"].size(0)
            correct += (output.argmax(1) == data["materials"]).sum().item()
    model.train()

    val_loss = total_loss / total
    accuracy = correct / total
    print(f"Validation Loss: {val_loss}, Accuracy: {accuracy}")
    if log_wandb:
        wandb.log({"val/loss": val_loss, "val/accuracy": accuracy}, step)

    return total_loss / total, correct / total


def main(log_name="finetuning"):
    global inputs_common

    train_loader, val_loader = create_dataloaders(root_dir="/home/GreatestHits/vis-data-256", batch_size=batch_size,
                                                  val_ratio=eval_ratio)

    bs = train_loader.batch_size
    all_material_names = train_loader.dataset.dataset.all_material_names
    num_classes = len(all_material_names)
    lr = 3e-4

    model = ImageAudioModel(num_classes=num_classes)

    if load_filename:
        model.load_state_dict(torch.load(load_filename))
    # model = model.to(device)

    if freeze_encoders:
        for param in model.image_encoder.parameters():
            param.requires_grad = False
        for param in model.audio_encoder.parameters():
            param.requires_grad = False

    if log_wandb:
        wandb.init(project="material-estimation", name=log_name, config={
            "epochs": num_epochs,
            "batch_size": bs,
            "lr": lr,
            "freeze_encoders": freeze_encoders,
            "image_encoder": model.image_encoder_name,
            "audio_encoder": model.audio_encoder_name,
            "dataset": "GreatestHits",
        })

    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
    inputs_common = {
        'image_starts': torch.tensor([tokenizer('<image>')['input_ids']] * bs, dtype=torch.int),
        'image_ends': torch.tensor([tokenizer('</image>')['input_ids']] * bs, dtype=torch.int),
        'audio_starts': torch.tensor([tokenizer('<audio>')['input_ids']] * bs, dtype=torch.int),
        'audio_ends': torch.tensor([tokenizer('</audio>')['input_ids']] * bs, dtype=torch.int),
        'input_ids': torch.tensor([tokenizer('<text>')['input_ids']] * bs, dtype=torch.int),
        'input_ide': torch.tensor([tokenizer('</text>')['input_ids']] * bs, dtype=torch.int)
    }
    inputs_common = {k: inputs_common[k].to(device) for k in inputs_common}

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    loss_values = []
    prev_time = time.time()
    pbar = tqdm(total=num_epochs * len(train_loader), desc="Training started")
    step = 0
    curr_epoch = 0
    if load_filename:
        curr_epoch = int(load_filename.split(".")[0].split("_")[1])
        pbar.update(curr_epoch * len(train_loader))
    for epoch in range(curr_epoch, num_epochs):
        for i, data in enumerate(train_loader):
            data = {k: data[k].to(device) for k in data}

            optimizer.zero_grad()
            output = model(data, inputs_common)
            loss = criterion(output, data["materials"])
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())

            if log_wandb:
                wandb.log({"train/loss": loss.item()}, step)
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item():.4f}")

            # Saving progress every 15 mins
            curr_time = time.time()
            if curr_time - prev_time > 900:
                prev_time = curr_time
                # print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}")
                torch.save(model.state_dict(), f"model_{epoch}_{i}.pth")
                torch.save(loss_values, "loss_history.pth")

            if i % 5000 == 0:
                evaluate(model, val_loader, step)
            step += 1

    torch.save(model.state_dict(), "final_model.pth")
    torch.save(loss_values, "loss_history.pth")

    return model


if __name__ == '__main__':
    # dummy_test()

    experiment_name = "partial_finetuning"
    batch_size = 1
    num_epochs = 10
    eval_ratio = 0.01
    load_filename = None
    log_wandb = True
    freeze_encoders = True

    main(experiment_name)
