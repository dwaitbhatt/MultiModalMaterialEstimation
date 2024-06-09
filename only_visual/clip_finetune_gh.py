import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import json
import wandb

class GreatestHitsDataset(Dataset):
    def __init__(self, root_path):
        self.root = root_path
        self.dataset_cache_dir = root_path + "-processed"
        # self.all_frames = os.listdir(self.root.joinpath(f"{self.train}_frames"))
        # with open(self.root.joinpath(f"{self.train}_labels.pkl"), "rb") as f:
        #     self.labels_map = pickle.load(f)  
        with open(f"{self.dataset_cache_dir}/times_info.json", "r") as f:
            self.times_info = json.load(f)
        self.all_classes = all_classes
        
    def __len__(self):
        # return len(self.all_frames)
        total_len = 0
        for key in self.times_info.keys():
            total_len += len(self.times_info[key])
        return total_len
    
    def __getitem__(self, idx):
        # fname = self.all_frames[idx]
        # image_path = self.root.joinpath(f"{self.train}_frames", fname)

        for key in self.times_info.keys():
            if idx < len(self.times_info[key]):
                break
            else:
                idx -= len(self.times_info[key])
        
        date_time = key
        frames_info = self.times_info[date_time]
        frame_timestamp = frames_info[idx][0]
        material_name = frames_info[idx][1]
        
        image_path = f"{self.dataset_cache_dir}/{date_time}_{frame_timestamp}_frame.jpg"
        # label = self.labels_map[fname.split(".")[0]]
        label = material_name
        inputs = processor(images=Image.open(image_path), return_tensors="pt")
        
        return inputs, self.all_classes.index(label)


def freeze_encoders(model):    
    # freezing the encoders
    for params in model.text_model.children():
        params.requires_grad = False
        
    for params in model.vision_model.children():
        params.requires_grad = False
    return model

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is None: continue
        p.grad.data = p.grad.data.float()


def create_dataloaders(root_path="../vis-data-256"):
    # train_dataset = GreatestHitsDataset("../vis-data-256", train=True)
    # val_dataset = GreatestHitsDataset("../vis-data-256", train=False)
    # train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    dataset = GreatestHitsDataset(root_path)

    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    return train_dataloader, val_dataloader


def generate_text_inputs():
    texts = list(map(lambda x: f"a drumstick is hitting an object made of {x}", all_classes))
    text_input = processor(text = texts, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
    return text_input


def prepare_input(image, text, label):
    return {
        'input_ids': text['input_ids'][label].to(device), 
        'attention_mask': text['attention_mask'][label].to(device),
        'pixel_values': image['pixel_values'].squeeze(1).to(device)
    }


def evaluate(model, dataloader, texts, losses):
    loss_img, loss_txt = losses
    val_loss = 0
    true = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            curr_image, label = batch
            model_input = prepare_input(curr_image, texts, label)
            output = model(**model_input)
            logits_per_image, logits_per_text = output.logits_per_image, output.logits_per_text
            n = logits_per_image.shape[0]

            # Compute loss
            ground_truth = torch.arange(n,dtype=torch.long,device=device)
            loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            val_loss += loss.item()

            # Compute accuracy
            vf = output["image_embeds"]
            tf = output["text_embeds"]
            probs = torch.softmax(vf @ tf.T, dim=1)
            max_probs = probs.argmax(dim=1)
            true += torch.sum(max_probs == label).item()
            total += n
        
        avg_val_loss = val_loss / len(dataloader)
        avg_val_acc = true / total
        return avg_val_acc, avg_val_loss


def train():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    
    if freeze_encoder:
        model = freeze_encoders(model)
    text_input = generate_text_inputs().to(device)
    
    convert_models_to_fp32(model)
    
    
    train_dataloader, val_dataloader = create_dataloaders()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            curr_image, label = batch
            model_input = prepare_input(curr_image, text_input, label)

            output = model(**model_input)
            logits_per_image, logits_per_text = output.logits_per_image, output.logits_per_text
            n = logits_per_image.shape[0]

            # Compute loss
            ground_truth = torch.arange(n,dtype=torch.long,device=device)
            loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            
            wandb.log({"train/loss": loss.item()})
            
        
            if i % eval_freq == 0:
                avg_val_acc, avg_val_loss = evaluate(model, val_dataloader, text_input, (loss_img, loss_txt))
                wandb.log({"epoch": epoch, "val/loss": avg_val_loss, "val/acc": avg_val_acc})
                if best_val_loss > avg_val_loss:
                    best_val_loss = avg_val_loss
                    model.save_pretrained(Path(SAVE_MODEL_DIR))
                
                model.train()

    final_comparision(val_dataloader)
     
        
def final_comparision(val_dataloader):
    base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    base_model.to(device)

    ft_model = CLIPModel.from_pretrained(Path(SAVE_MODEL_DIR))
    ft_model.to(device)

    text_input = generate_text_inputs()
    
    no_loss = lambda x, y: 0
    
    base_val_acc, _ = evaluate(base_model, val_dataloader, text_input,(no_loss, no_loss))
    ft_val_acc, _ = evaluate(ft_model, val_dataloader, text_input,(no_loss, no_loss))
    
    print(f"Base Model Accuracy: {base_val_acc}, \nFine-Tuned Model Accuracy: {ft_val_acc}")
    
if __name__ == "__main__":
    batch_size = 4
    val_ratio = 0.05
    num_epochs = 5
    freeze_encoder = False
    SAVE_MODEL_DIR = "gh_model"
    learning_rate = 1e-5
    eval_freq = 1000

    wandb.init(
        # set the wandb project where this run will be logged
        project="material-estimation",
        name="clip_finetune_full",
        # track hyperparameters and run metadata
        config={
            "architecture": "CLIP",
            "dataset": "greatest-hits",
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "freeze_encoder": freeze_encoder,
            "learning_rate": learning_rate,
        }
    )

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
        
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
