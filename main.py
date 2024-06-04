from transformers import WhisperConfig, WhisperModel, CLIPConfig, CLIPModel, AutoTokenizer
import whisper
import torch
import torch.nn as nn
from dataset_utils import _transform, create_dataloader
import time
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn


class TextClassificationModel(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(TextClassificationModel, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.fc = nn.Linear(hidden_size, num_classes)   # Fully connected layer

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
        self.whisper_config = WhisperConfig()
        self.audio_encoder = WhisperModel(self.whisper_config).to(device)
        self.image_config = CLIPConfig()
        self.image_encoder = CLIPModel(self.image_config).to(device)
        self.audio_align_attention = nn.MultiheadAttention(256,
                                                      4 * 2,
                                                      dropout=0,
                                                      add_bias_kv=False,
                                                      add_zero_attn=False).to(device)

        self.image_align_attention = nn.MultiheadAttention(256,
                                                      4 * 2,
                                                      dropout=0,
                                                      add_bias_kv=False,
                                                      add_zero_attn=False).to(device)

        self.transform_audio_to_hidden = nn.Linear(256,
                                              256).to(device)
        self.transform_image_to_hidden = nn.Linear(512,
                                              256).to(device)

        self.project_image = nn.Conv1d(512, 512,
                                  kernel_size=3, stride=1).to(device)

        self.project_audio = nn.Conv1d(256, 256,
                                  kernel_size=3, stride=1).to(device)

        self.embed_tokens = nn.Embedding(self.whisper_config.vocab_size, 256).to(device)
        self.num_classes = num_classes
        self.final_classifier = TextClassificationModel(256, num_classes).to(device)

    def forward(self, inputs, inputs_common):
        image_features = self.image_encoder.visual_projection(self.image_encoder.vision_model(inputs['images']).last_hidden_state)
        audio_features = self.audio_encoder.encoder(inputs['audios']).last_hidden_state

        text_embeddings = self.embed_tokens(inputs_common['input_ids'])

        token_embeddings = self.embed_tokens.weight.unsqueeze(0).repeat(
            text_embeddings.size(0), 1, 1).transpose(0, 1)

        audio_starts = self.embed_tokens(inputs_common['audio_starts'])
        audio_ends = self.embed_tokens(inputs_common['audio_ends'])

        audio_features = self.project_audio(audio_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        audio_features = self.transform_audio_to_hidden(audio_features)

        audio_features = self.audio_align_attention(audio_features.transpose(0, 1),
                                               token_embeddings, token_embeddings)[0].transpose(0, 1).contiguous()

        audio_inputs = torch.cat([torch.cat([audio_starts, audio_features], dim=1), audio_ends], dim=1)
        text_embeddings = torch.cat([text_embeddings, audio_inputs], dim=1)

        image_starts = self.embed_tokens(inputs_common['image_starts'])
        image_ends = self.embed_tokens(inputs_common['image_ends'])

        image_features = self.project_image(image_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

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
    all_audio_mels = mel.unsqueeze(0) # (1, 80, 3000)

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
                        'input_ids' : torch.tensor([tokenizer('<text>')['input_ids']] * bs, dtype=torch.int),
                        'input_ide': torch.tensor([tokenizer('</text>')['input_ids']] * bs, dtype=torch.int)
                    }
    inputs = {'audios': all_audio_mels, 'images': frame}
    
    inputs = {k: inputs[k].to(device) for k in inputs}
    inputs_common = {k: inputs_common[k].to(device) for k in inputs_common}
    
    model = ImageAudioModel(num_classes=30)
    model = model.to(device)
    output = model(inputs, inputs_common)
    print(output.shape)  # Should be [batch_size, num_classes]


def train_model(dataloader, num_epochs=10):
    bs = dataloader.batch_size
    mat_to_ind = dataloader.dataset.mat_to_ind
    num_classes = len(mat_to_ind)
    model = ImageAudioModel(num_classes=num_classes)
    model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
    inputs_common = {
                        'image_starts': torch.tensor([tokenizer('<image>')['input_ids']] * bs, dtype=torch.int),
                        'image_ends': torch.tensor([tokenizer('</image>')['input_ids']] * bs, dtype=torch.int),
                        'audio_starts': torch.tensor([tokenizer('<audio>')['input_ids']] * bs, dtype=torch.int),
                        'audio_ends': torch.tensor([tokenizer('</audio>')['input_ids']] * bs, dtype=torch.int),
                        'input_ids' : torch.tensor([tokenizer('<text>')['input_ids']] * bs, dtype=torch.int),
                        'input_ide': torch.tensor([tokenizer('</text>')['input_ids']] * bs, dtype=torch.int)
                    }
    inputs_common = {k: inputs_common[k].to(device) for k in inputs_common}    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()
    
    loss_values = []
    prev_time = time.time()
    pbar = tqdm(total=num_epochs*len(dataloader), desc="Training started")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            data = {k: data[k].to(device) for k in data}    
            
            optimizer.zero_grad()
            output = model(data, inputs_common)
            loss = criterion(output, data["materials"])
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}")
            
            # Saving progress every 5 mins
            curr_time = time.time()
            if curr_time - prev_time > 300:
                prev_time = curr_time
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}")
                torch.save(model.state_dict(), f"model_{epoch}_{i}.pth")
                torch.save(loss_values, "loss_history.pth")

    torch.save(model.state_dict(), "final_model.pth")
    torch.save(loss_values, "loss_history.pth")
    
    return model

if __name__ == '__main__':
    # dummy_test()
    
    batch_size = 1
    dataloader = create_dataloader(root_dir="/home/GreatestHits/vis-data-256", batch_size=batch_size)
    
    train_model(dataloader)