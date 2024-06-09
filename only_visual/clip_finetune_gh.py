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
import pickle
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
        "dataset": "greatest-hits",
    }
)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
        



class GreatestHitsDataset(Dataset):
    def __init__(self, root_path, train=True):
        self.root = root_path
        self.train = "train" if train else "test"
        self.all_frames = os.listdir(self.root.joinpath(f"{self.train}_frames"))
        with open(self.root.joinpath(f"{self.train}_labels.pkl"), "rb") as f:
            self.labels_map = pickle.load(f)  
        self.all_classes = \
        ['paper',
         'plastic-bag',
         'leaf',
         'tile',
         'drywall',
         'glass',
         'ceramic',
         'plastic',
         'grass',
         'carpet',
         'metal',
         'dirt',
         'water',
         'wood',
         'gravel',
         'rock',
         'cloth'
        ]
        
    def __len__(self):
        return len(self.all_frames)
    
    def __getitem__(self, idx):
        fname = self.all_frames[idx]
        image_path = self.root.joinpath(f"{self.train}_frames", fname)
        label = self.labels_map[fname.split(".")[0]]
        inputs = processor(images=Image.open(image_path), return_tensors="pt")
        
        return inputs, self.all_classes.index(label[1])
    
all_classes = \
        ['paper',
         'plastic-bag',
         'leaf',
         'tile',
         'drywall',
         'glass',
         'ceramic',
         'plastic',
         'grass',
         'carpet',
         'metal',
         'dirt',
         'water',
         'wood',
         'gravel',
         'rock',
         'cloth'
        ]


model.to(device)

train_dataset = GreatestHitsDataset(Path("GreatestHits"), train=True)
val_dataset = GreatestHitsDataset(Path("GreatestHits"), train=False)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

texts = list(map(lambda x: f"a drumstick is hitting an object made of {x}", all_classes))
text_input = processor(text = texts, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
text_input.to(device)
convert_models_to_fp32(model)

num_epochs = 5
best_val_loss = float('inf')

for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()

        curr_input, label = batch
        tf = {'input_ids': text_input['input_ids'][label].to(device), 
              'attention_mask': text_input['attention_mask'][label].to(device)
             }
        vi = curr_input['pixel_values'].squeeze(1).to(device)
        output = model(pixel_values=vi, **tf)
        logits_per_image, logits_per_text = output.logits_per_image, output.logits_per_text
        n = logits_per_image.shape[0]

        # Compute loss
        ground_truth = torch.arange(n,dtype=torch.long,device=device)
        loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        loss.backward()
        total_loss += loss.item()
        
        optimizer.step()
    
    
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            
            curr_input, label = batch
            tf = {'input_ids': text_input['input_ids'][label].to(device), 
                  'attention_mask': text_input['attention_mask'][label].to(device)
                 }
            vi = curr_input['pixel_values'].squeeze(1).to(device)
            output = model(pixel_values=vi, **tf)
            logits_per_image, logits_per_text = output.logits_per_image, output.logits_per_text
            n = logits_per_image.shape[0]

            # Compute loss
            ground_truth = torch.arange(n,dtype=torch.long,device=device)
            loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_dataloader)
    avg_train_loss = total_loss / len(train_dataloader)
    wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
    if best_val_loss > avg_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(Path("gh_model"))
        
    
del model

base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

ft_model = CLIPModel.from_pretrained(Path("gh_model"))
base_model.to(device)
ft_model.to(device)

base_tf = base_model.get_text_features(**text_input)
ft_tf = ft_model.get_text_features(**text_input)
base_tf.to(device)
ft_tf.to(device)

base_true = 0
ft_true = 0

with torch.no_grad():
    for batch in val_dataloader:
        curr_input, label = batch
        label = label.to(device)
        tf = {'input_ids': text_input['input_ids'][label].to(device), 
              'attention_mask': text_input['attention_mask'][label].to(device)
             }
        vi = curr_input['pixel_values'].squeeze(1).to(device)
        bvf = base_model.get_image_features(pixel_values=vi)
        ftvf = ft_model.get_image_features(pixel_values=vi)
        
        base_probs = torch.softmax(bvf @ base_tf.T, dim=1)
        ft_probs = torch.softmax(ftvf @ ft_tf.T, dim=1)
        
        max_b_probs = base_probs.argmax(dim=1)
        max_ft_probs = ft_probs.argmax(dim=1)
        
        base_true += torch.sum(max_b_probs == label).item()
        ft_true += torch.sum(max_ft_probs == label).item()
        
print(f"Base Model Accuracy: {base_true / 200}, Fine-Tuned Model Accuracy: {ft_true / 200}")