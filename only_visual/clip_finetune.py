import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
from pathlib import Path
import clip
from tqdm import tqdm

import wandb
import os

os.environ["WANDB_API_KEY"] = "YOUR_API_KEY"

wandb.init(
    # set the wandb project where this run will be logged
    project="clip-finetune",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 1e-5,
        "architecture": "CLIP",
        "dataset": "flickr-materials",
        "epochs": 10,
    }
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

all_classes = os.listdir(Path("flickr_materials", "image"))
texts = list(map(lambda x: f"a picture of an object made of {x}", all_classes))


class CustomDataset(Dataset):
    def __init__(self):
        self.basepath = Path("flickr_materials", "image")
        self.classes = os.listdir(self.basepath)
        self.all_images = []
        for idx, nam in enumerate(self.classes):
            self.all_images.extend(os.listdir(self.basepath.joinpath(nam)))
            
        
    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        curr_label = idx // 100
        image_path = self.basepath.joinpath(self.classes[curr_label], self.all_images[idx])
        curr_text = texts[curr_label]
        curr_input = processor(images=Image.open(image_path), text=curr_text, return_tensors="pt", padding=True)
        curr_input = {k:v.squeeze().to(device) for k, v in curr_input.items()}
        
        return curr_input, curr_label

# freezing the encoders
for params in model.text_model.children():
    params.requires_grad = False
    
for params in model.vision_model.children():
    params.requires_grad = False
    
    
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is None: continue
        p.grad.data = p.grad.data.float()
        

model.to(device)

BATCH_SIZE = 4
LR = 1e-5
dataset = CustomDataset()
train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=LR,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

convert_models_to_fp32(model)

num_epochs = 10
best_val_loss = float('inf')
saved_models = []

for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()

        curr_input, _ = batch
        output = model(**curr_input)
        logits_per_image, logits_per_text = output.logits_per_image, output.logits_per_text

        # Compute loss
        n = curr_input['input_ids'].shape
        ground_truth = torch.arange(logits_per_image.shape[0],dtype=torch.long,device=device)
        loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            
            curr_input, _ = batch
            output = model(**curr_input)
            logits_per_image, logits_per_text = output.logits_per_image, output.logits_per_text

            # Compute loss
            ground_truth = torch.arange(logits_per_image.shape[0],dtype=torch.long,device=device)
            loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_dataloader)
    avg_train_loss = total_loss / len(train_dataloader)
    wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
    if best_val_loss > val_loss:
        best_val_loss = val_loss
        model.save_pretrained(Path("model"))
        

        
del model

base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

ft_model = CLIPModel.from_pretrained(Path("model"))
base_model.to(device)
ft_model.to(device)

texts = list(map(lambda x: f"a picture of an object made of {x}", all_classes))
text_input = processor(text = texts, return_tensors="pt", padding=True)
text_input.to(device)
base_tf = base_model.get_text_features(**text_input)
ft_tf = ft_model.get_text_features(**text_input)
base_tf.to(device)
ft_tf.to(device)

base_true = 0
ft_true = 0

with torch.no_grad():
    for batch in val_dataloader:
        curr_input, lab = batch
        lab = lab.to(device)
        vi = curr_input['pixel_values']
        vf = base_model.get_image_features(pixel_values=vi)
        
        base_probs = torch.softmax(vf @ base_tf.T, dim=1)
        ft_probs = torch.softmax(vf @ ft_tf.T, dim=1)
        
        max_b_probs = base_probs.argmax(dim=1)
        max_ft_probs = ft_probs.argmax(dim=1)
        
        base_true += torch.sum(max_b_probs == lab).item()
        ft_true += torch.sum(max_ft_probs == lab).item()
        
print(f"Base Model Accuracy: {base_true / 200}, Fine-Tuned Model Accuracy: {ft_true / 200}")