from transformers import WhisperModel, CLIPModel, AutoTokenizer, AutoModelForCausalLM
import whisper
import torch
import torch.nn as nn
from dataset_utils import _transform, create_dataloaders
from tqdm.auto import tqdm
import time
import pathlib
import wandb
import os
import json
import argparse
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class MaterialEstimationModel(nn.Module):
    def __init__(self, num_classes, freeze_encoders=True):
        super(MaterialEstimationModel, self).__init__()
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

        self.freeze_encoders = freeze_encoders
        if self.freeze_encoders:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

        self.image_h_dim = 512
        self.audio_h_dim = 512

        self.image_seq_len = 50
        self.audio_seq_len = 1500
        
        self.aligned_seq_len = 32
    
        llm = AutoModelForCausalLM.from_pretrained(self.llm_name)
        self.embed_tokens = llm.model.embed_tokens.to(device)
        self.llm_dim = self.embed_tokens.weight.shape[1]

        self.project_image = nn.Conv1d(self.image_seq_len, self.aligned_seq_len,
                                       kernel_size=1, stride=1).to(device)

        self.project_audio = nn.Conv1d(self.audio_seq_len, self.aligned_seq_len,
                                       kernel_size=1, stride=1).to(device)

        self.transform_image_to_hidden = nn.Linear(self.image_h_dim,
                                                   self.llm_dim).to(device)
        self.transform_audio_to_hidden = nn.Linear(self.audio_h_dim,
                                                   self.llm_dim).to(device)
        
        self.image_align_attention = nn.MultiheadAttention(self.llm_dim,
                                                           4 * 2,
                                                           dropout=0,
                                                           add_bias_kv=False,
                                                           add_zero_attn=False).to(device)
        self.audio_align_attention = nn.MultiheadAttention(self.llm_dim,
                                                           4 * 2,
                                                           dropout=0,
                                                           add_bias_kv=False,
                                                           add_zero_attn=False).to(device)

        self.num_classes = num_classes
        self.final_classifier = TextClassificationModel(self.llm_dim, num_classes).to(device)

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

    def get_relevant_modules(self):
        image_nns = []
        audio_nns = []
        if not self.freeze_encoders:
            image_nns.append(self.image_encoder)
            audio_nns.append(self.audio_encoder)
        image_nns.extend([self.project_image, self.transform_image_to_hidden, self.image_align_attention])
        audio_nns.extend([self.project_audio, self.transform_audio_to_hidden, self.audio_align_attention])
        return nn.ModuleList(image_nns + audio_nns + [self.final_classifier])

    def weights_to_save(self):
        modules_to_save = self.get_relevant_modules()
        return modules_to_save.state_dict()

    def load_weights(self, weights_to_load):
        modules_to_load = self.get_relevant_modules()
        modules_to_load.load_state_dict(weights_to_load)

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

    model = MaterialEstimationModel(num_classes=30)
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
            if i == len(dataloader) - 1:
                break
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

    train_loader, val_loader = create_dataloaders(root_dir="./vis-data-256", batch_size=batch_size,
                                                  val_ratio=eval_ratio)

    bs = train_loader.batch_size
    all_material_names = train_loader.dataset.dataset.all_material_names
    num_classes = len(all_material_names)
    lr = 3e-4

    model = MaterialEstimationModel(num_classes=num_classes, freeze_encoders=freeze_encoders)
    # print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    # model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Training with batch size {batch_size} across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        model_ref = model.module
    else:
        model_ref = model
    image_encoder_name = model_ref.image_encoder_name
    audio_encoder_name = model_ref.audio_encoder_name
    model.train()

    if log_wandb:
        wandb.init(project="material-estimation", name=log_name, config={
            "epochs": num_epochs,
            "batch_size": bs,
            "lr": lr,
            "freeze_encoders": freeze_encoders,
            "image_encoder": image_encoder_name,
            "audio_encoder": audio_encoder_name,
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

    pbar = tqdm(total=num_epochs * len(train_loader), desc="Training started")
    step = 0
    curr_epoch = 0
    if ckpt_path:
        curr_epoch = int(ckpt_path.split("/")[-1].split(".")[0].split("_")[2])
        pbar.update(curr_epoch * len(train_loader))
    for epoch in range(curr_epoch, num_epochs):
        for i, data in enumerate(train_loader):
            if i == len(train_loader) - 1:
                break
            data = {k: data[k].to(device) for k in data}

            optimizer.zero_grad()
            output = model(data, inputs_common)
            loss = criterion(output, data["materials"])
            loss.backward()
            optimizer.step()

            if log_wandb:
                wandb.log({"train/loss": loss.item()}, step)
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item():.4f}")

            if i % 5000 == 0:
                checkpoint = {  
                                'model_state_dict': model_ref.weights_to_save(),
                                'optimizer_state_dict': optimizer.state_dict()
                             }
                val_loss, val_acc = evaluate(model, val_loader, step)
                print(f"Validation Loss: {val_loss}, Accuracy: {val_acc}")
                print(f"Saving model at {save_dir + f'model_ckpt_{epoch}_{i}_{val_acc:.4f}.pth'}")
                torch.save(checkpoint, save_dir + f"model_ckpt_{epoch}_{i}_{val_acc:.4f}.pth")
            step += 1
        gc.collect()
        torch.cuda.empty_cache()

    checkpoint = {  
        'model_state_dict': model_ref.weights_to_save(),
        'optimizer_state_dict': optimizer.state_dict()
     }
    torch.save(checkpoint, save_dir + "final_model_ckpt.pth")
    # torch.save(loss_values, "loss_history.pth")

    return model


if __name__ == '__main__':
    # dummy_test()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config file")
    args = parser.parse_args()
    
    try:
        with open(args.config) as f:
            config_params = json.load(f)
            experiment_name = config_params.get("experiment_name", "alignment_training_cached")
            batch_size = config_params.get("batch_size", 4)
            num_epochs = config_params.get("num_epochs", 10)
            eval_ratio = config_params.get("eval_ratio", 0.05)
            ckpt_path = config_params.get("ckpt_path", None)
            log_wandb = config_params.get("log_wandb", True)
            freeze_encoders = config_params.get("freeze_encoders", True)
    except FileNotFoundError:
        print(f"{args.config} file not found. Please make sure the file exists in the current directory.")
        exit(1)

    batch_size *= torch.cuda.device_count()

    save_dir = pathlib.Path(__file__).resolve().parent
    save_dir = save_dir / 'ckpts' / time.strftime("%m.%d.%Y")
    save_subdir = '_'.join([
        time.strftime("%H-%M-%S"),
        experiment_name
    ])
    save_dir = save_dir / save_subdir
    save_dir = str(save_dir) + "/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    main(experiment_name)
