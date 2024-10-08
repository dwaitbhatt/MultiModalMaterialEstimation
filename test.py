import argparse
import json
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from dataset_utils import create_dataloaders
from train import MaterialEstimationModel
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from collections import Counter


def performance_metrics(all_labels, all_preds, all_material_names):
    print("Classification report for material estimation:")
    print(classification_report(all_labels, all_preds, target_names=all_material_names, labels=np.arange(num_classes)))

    print(f"Number of samples: {len(all_labels)}")
    print(f"Unique labels: {set(all_labels)}")
    print(f"Unique predictions: {set(all_preds)}")

    print("Most common mistakes per class:")
    for true_class_index, true_class in enumerate(all_material_names):
        true_class_samples = [pred for true, pred in zip(all_labels, all_preds) if true == true_class_index]
        misclassified = [pred for pred in true_class_samples if pred != true_class_index]
        if misclassified:
            misclassified_frequencies = Counter(misclassified)
            most_common_mistakes = misclassified_frequencies.most_common(2)
            print(f"{true_class}:")
            for mistake, count in most_common_mistakes:
                print(f"    {all_material_names[mistake]} ({count} times)")
        else:
            if true_class_samples:
                print(f"{true_class}: No misclassifications")
            else:
                print(f"{true_class}: No samples")


def test(model, dataloader):
    model.eval()
    
    try:
        all_material_names = dataloader.dataset.all_material_names
    except AttributeError:
        all_material_names = dataloader.dataset.dataset.all_material_names

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader),
                         total=len(dataloader),
                         leave=False,
                         desc="Evaluating on test data"):
            if i == len(dataloader) - 1:
                break
            data = {k: data[k].to(device) for k in data}
            output = model(data, inputs_common)

            preds = output.argmax(1).cpu().numpy()
            labels = data["materials"].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    cm = confusion_matrix(all_labels, all_preds)

    # performance_metrics(all_labels, all_preds, all_material_names)
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"\nAverage accuracy on test data: {accuracy*100:.2f} %")

    
def main():    
    global inputs_common, num_classes
    dataloader, _ = create_dataloaders(root_dir=data_path, batch_size=batch_size, test=True)
    # _, dataloader = create_dataloaders(root_dir=data_path, batch_size=batch_size, val_ratio=0.01)

    bs = dataloader.batch_size
    num_classes = 17
    
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
    
    model = MaterialEstimationModel(num_classes=num_classes, freeze_encoders=True)
    ckpt = torch.load(ckpt_path)
    model.load_weights(ckpt['model_state_dict'])

    if torch.cuda.device_count() > 1:
        print(f"Testing with batch size {batch_size} across {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    test(model, dataloader)
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config file", default="config_test.json")
    args = parser.parse_args()

    try:
        with open(args.config) as f:
            config_params = json.load(f)
            experiment_name = config_params.get("experiment_name", "alignment_training_cached")
            batch_size = config_params.get("batch_size", 4)
            ckpt_path = config_params.get("ckpt_path", None)
            freeze_encoders = config_params.get("freeze_encoders", True)
            data_path = config_params.get("data_path", "./vis-data-256")
    except FileNotFoundError:
        print(f"{args.config} file not found. Please make sure the file exists in the current directory.")
        exit(1)
        
    if ckpt_path is None:
        print("Please provide the path to the model checkpoint file in the config ckpt_path parameter.")
        exit(1)
    batch_size *= torch.cuda.device_count()

    main()