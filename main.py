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

if __name__ == '__main__':
    # whisper_config = WhisperConfig.from_pretrained('trained_models/whisper_model')
    whisper_config = WhisperConfig()
    audio_encoder = WhisperModel(whisper_config).to(device)
    audio = whisper.load_audio(".\\aud.wav")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    all_audio_mels = mel.unsqueeze(0) # (1, 80, 3000)

    preprocess = _transform(224)
    frame = preprocess(Image.open(".\\img.JPG"))
    frame = frame.unsqueeze(0)
    image_config = CLIPConfig()
    image_encoder = CLIPModel(image_config).to(device)

    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
    bs = 1

    # inputs = {'audios': all_audio_mels.half()}
    inputs = {'audios': all_audio_mels, 'images': frame,
              'image_starts': torch.tensor([tokenizer('<image>')['input_ids']] * bs, dtype=torch.int),
              'image_ends': torch.tensor([tokenizer('</image>')['input_ids']] * bs, dtype=torch.int),
              'audio_starts': torch.tensor([tokenizer('<audio>')['input_ids']] * bs, dtype=torch.int),
              'audio_ends': torch.tensor([tokenizer('</audio>')['input_ids']] * bs, dtype=torch.int),
              'input_ids' : torch.tensor([tokenizer('<text>')['input_ids']] * bs, dtype=torch.int),
              'input_ide': torch.tensor([tokenizer('</text>')['input_ids']] * bs, dtype=torch.int)
              }

    inputs = {k: inputs[k].to(device) for k in inputs}

    image_features = image_encoder.visual_projection(image_encoder.vision_model(inputs['images']).last_hidden_state)
    # torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
    # (1, 50, 512)
    audio_features = audio_encoder.encoder(inputs['audios']).last_hidden_state
    # torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
    # (1, 1500, 256)

    audio_align_attention = nn.MultiheadAttention(256,
                                                       4 * 2,
                                                       dropout=0,
                                                       add_bias_kv=False,
                                                       add_zero_attn=False).to(device)

    image_align_attention = nn.MultiheadAttention(256,
                                                       4 * 2,
                                                       dropout=0,
                                                       add_bias_kv=False,
                                                       add_zero_attn=False).to(device)

    transform_audio_to_hidden = nn.Linear(256,
                                          256).to(device)
    transform_image_to_hidden = nn.Linear(512,
                                          256).to(device)

    project_image = nn.Conv1d(512, 512,
                                   kernel_size=3, stride=1).to(device)

    project_audio = nn.Conv1d(256, 256,
                                   kernel_size=3, stride=1).to(device)


    embed_tokens = nn.Embedding(whisper_config.vocab_size, 256).to(device)

    text_embeddings = embed_tokens(inputs['input_ids'])  # (1,3,256)

    token_embeddings = embed_tokens.weight.unsqueeze(0).repeat(
        text_embeddings.size(0), 1, 1).transpose(0, 1)

    audio_starts = embed_tokens(inputs['audio_starts'])
    audio_ends = embed_tokens(inputs['audio_ends'])

    audio_features = project_audio(audio_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
    audio_features = transform_audio_to_hidden(audio_features)

    audio_features = audio_align_attention(audio_features.transpose(0, 1),
                                                token_embeddings, token_embeddings)[0].transpose(0, 1).contiguous()


    audio_inputs = torch.cat([torch.cat([audio_starts, audio_features], dim=1), audio_ends], dim=1)
    # (1,1504,256)

    text_embeddings = torch.cat([text_embeddings, audio_inputs], dim=1)
    # torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
    # (1, 1507, 256)

    image_starts = embed_tokens(inputs['image_starts'])
    image_ends = embed_tokens(inputs['image_ends'])

    image_features = project_image(image_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

    image_features = transform_image_to_hidden(image_features)
    image_features = image_align_attention(image_features.transpose(0, 1),
                                                token_embeddings, token_embeddings)[0].transpose(0, 1).contiguous()

    image_inputs = torch.cat([torch.cat([image_starts, image_features], dim=1), image_ends], dim=1)

    text_embeddings = torch.cat(
        [torch.cat([text_embeddings, image_inputs], dim=1),
         embed_tokens(inputs['input_ide'])], dim=1)

    # torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
    # (1, 1564, 256)

    batch_size = 1
    sequence_length = text_embeddings.shape[1]
    hidden_size = text_embeddings.shape[2]
    num_classes = 30

    model = TextClassificationModel(hidden_size, num_classes)
    input_tensor = text_embeddings
    output = model(input_tensor)
    print(output.shape)  # Should be [batch_size, num_classes]
