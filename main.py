import pdb
from transformers import WhisperConfig, WhisperModel, CLIPConfig, CLIPModel, AutoTokenizer
import whisper
import torch
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import torch.nn as nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def _transform(n_px):
    return Compose([
        Resize(n_px),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _convert_image_to_rgb(image):
    return image.convert("RGB")

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



    def forward(self, inputs):
        image_features = self.image_encoder.visual_projection(self.image_encoder.vision_model(inputs['images']).last_hidden_state)
        audio_features = self.audio_encoder.encoder(inputs['audios']).last_hidden_state

        text_embeddings = self.embed_tokens(inputs['input_ids'])

        token_embeddings = self.embed_tokens.weight.unsqueeze(0).repeat(
            text_embeddings.size(0), 1, 1).transpose(0, 1)

        audio_starts = self.embed_tokens(inputs['audio_starts'])
        audio_ends = self.embed_tokens(inputs['audio_ends'])

        audio_features = self.project_audio(audio_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        audio_features = self.transform_audio_to_hidden(audio_features)

        audio_features = self.audio_align_attention(audio_features.transpose(0, 1),
                                               token_embeddings, token_embeddings)[0].transpose(0, 1).contiguous()

        audio_inputs = torch.cat([torch.cat([audio_starts, audio_features], dim=1), audio_ends], dim=1)
        text_embeddings = torch.cat([text_embeddings, audio_inputs], dim=1)

        image_starts = self.embed_tokens(inputs['image_starts'])
        image_ends = self.embed_tokens(inputs['image_ends'])

        image_features = self.project_image(image_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        image_features = self.transform_image_to_hidden(image_features)
        image_features = self.image_align_attention(image_features.transpose(0, 1),
                                               token_embeddings, token_embeddings)[0].transpose(0, 1).contiguous()

        image_inputs = torch.cat([torch.cat([image_starts, image_features], dim=1), image_ends], dim=1)

        text_embeddings = torch.cat(
            [torch.cat([text_embeddings, image_inputs], dim=1),
             self.embed_tokens(inputs['input_ide'])], dim=1)

        hidden_size = text_embeddings.shape[2]
        model = TextClassificationModel(hidden_size, self.num_classes)
        input_tensor = text_embeddings
        output = model(input_tensor)
        return output

if __name__ == '__main__':

    audio = whisper.load_audio(".\\aud.wav")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    all_audio_mels = mel.unsqueeze(0) # (1, 80, 3000)

    preprocess = _transform(224)
    frame = preprocess(Image.open(".\\img.JPG"))
    frame = frame.unsqueeze(0)

    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
    bs = 1

    inputs = {'audios': all_audio_mels, 'images': frame,
              'image_starts': torch.tensor([tokenizer('<image>')['input_ids']] * bs, dtype=torch.int),
              'image_ends': torch.tensor([tokenizer('</image>')['input_ids']] * bs, dtype=torch.int),
              'audio_starts': torch.tensor([tokenizer('<audio>')['input_ids']] * bs, dtype=torch.int),
              'audio_ends': torch.tensor([tokenizer('</audio>')['input_ids']] * bs, dtype=torch.int),
              'input_ids' : torch.tensor([tokenizer('<text>')['input_ids']] * bs, dtype=torch.int),
              'input_ide': torch.tensor([tokenizer('</text>')['input_ids']] * bs, dtype=torch.int)
              }

    inputs = {k: inputs[k].to(device) for k in inputs}

    model = ImageAudioModel(num_classes=30)
    output = model(inputs)
    print(output.shape)  # Should be [batch_size, num_classes]
