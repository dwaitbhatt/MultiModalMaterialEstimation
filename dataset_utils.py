from collections import OrderedDict
import whisper
import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split
# import torchaudio
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, ToPILImage, Normalize
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import json


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        ToPILImage(),
        Resize(n_px),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        # Normalization params from 1/6th of GreatestHits dataset
        Normalize((0.4738, 0.4298, 0.3702), 
                  (0.1994, 0.1913, 0.1805)),
    ])


class GreatestHitsDataset(Dataset):
    def __init__(self, root_dir="./vis-data-256", audio_length=5, use_cached=True, test=False):
        self.root_dir = root_dir
        self.transform = _transform(224)
        # self.transform_to_tensor = T.ToTensor()
        self.audio_rate = 96000
        self.audio_length = audio_length
        self.use_cached = use_cached
        self.dataset_cache_dir = root_dir + "-processed"
        
        if use_cached:
            # Load times_info from cached json
            with open(f"{self.dataset_cache_dir}/times_info.json", "r") as f:
                times_info = json.load(f)

            if test:
                with open(f"test_set_times_info.json", "r") as f:
                    self.times_info = json.load(f)
            else:
                self.times_info = times_info
        else:
            self.times_info = OrderedDict()
            for file in sorted(glob.glob(root_dir + "/*_times.txt")):
                with open(file, "r") as f:
                    data = f.readlines()
                    data = [line.strip().split() for line in data]
                    data = [(float(line[0]), line[1]) for line in data if (line[1] != "None") and (line[2] == "hit")]
                    self.times_info[file.split("/")[-1].split("_")[0]] = data
                            
        self.all_material_names = set()
        for _, frame_info in times_info.items():
            self.all_material_names.update([frame[1] for frame in frame_info])

        self.all_material_names = sorted(list(self.all_material_names))
        self.mat_to_ind = {name: i for i, name in enumerate(self.all_material_names)}
        self.ind_to_mat = {i: name for i, name in enumerate(self.all_material_names)}
    
    def __len__(self):
        total_len = 0
        for key in self.times_info.keys():
            total_len += len(self.times_info[key])
        return total_len
    
    def _dataset_idx_to_times_info_key_idx(self, idx):
        for key in self.times_info.keys():
            if idx < len(self.times_info[key]):
                break
            else:
                idx -= len(self.times_info[key])
        date_time = key
        return date_time, idx

    def __getitem__(self, idx):
        date_time, idx = self._dataset_idx_to_times_info_key_idx(idx)

        frames_info = self.times_info[date_time]
        frame_timestamp = frames_info[idx][0]
        material_name = frames_info[idx][1]
        material_index = self.mat_to_ind[material_name]
        material_index = torch.tensor(material_index)

        if self.use_cached:
            mel = np.load(f"{self.dataset_cache_dir}/{date_time}_{frame_timestamp}_mel.npy")
            mel = torch.tensor(mel)
            # Load jpg
            frame = cv2.imread(f"{self.dataset_cache_dir}/{date_time}_{frame_timestamp}_frame.jpg")
            frame = self.transform(frame)
        else:
            # frames, audio, metadata = torchvision.io.read_video(f"./vis-data-256/{date_time}_denoised.mp4")
            # Loading single frame with cv2 for faster frame access compared to loading whole video
            cap = cv2.VideoCapture(self.root_dir + f"/{date_time}_denoised.mp4")
            audio = whisper.load_audio(self.root_dir + f"/{date_time}_denoised.wav", self.audio_rate)

            audio_start_time = frame_timestamp - self.audio_length / 2
            audio_start_idx = int(audio_start_time * self.audio_rate)
            audio = audio[audio_start_idx : audio_start_idx + self.audio_rate * self.audio_length]
            audio = whisper.pad_or_trim(audio, self.audio_rate * self.audio_length)
            # print(date_time, frame_timestamp, audio.shape)
            mel = whisper.log_mel_spectrogram(audio)

            cap.set(cv2.CAP_PROP_POS_MSEC, frame_timestamp * 1000)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = (self.transform_to_tensor(frame) * 255).to(torch.uint8)
            frame = self.transform(frame)
            
            cap.release()
            
        # return {"images": frame, "audios": mel, "audios_raw":audio, "materials": material_index}
        return {"images": frame, "audios": mel, "materials": material_index}
    

# Create DataLoader
def create_dataloaders(root_dir="./vis-data-256", batch_size=4, val_ratio=0.05, use_cached=True, test=False):
    dataset = GreatestHitsDataset(root_dir, use_cached=use_cached, test=test)
    print(f"\n{'Test' if test else 'Training'} dataset size: {len(dataset)}\n")

    if test:
        train_set = dataset
        val_set = None
    else:
        val_size = int(val_ratio * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True) if val_set is not None else None 

    return train_loader, val_loader

    
def test_cached_vs_uncached_datasets(n_test=5, batch_size=4, compare_diff_images=False):
    print("Loading cached dataset...")
    start_time = time.time()
    cached_dataset = GreatestHitsDataset(root_dir=dataset_dir, use_cached=True)
    print(f"Time taken to initialize cached dataset: {time.time() - start_time:.2f} seconds")

    print("Loading uncached dataset...")
    start_time = time.time()
    uncached_dataset = GreatestHitsDataset(root_dir=dataset_dir, use_cached=False)
    print(f"Time taken to initialize uncached dataset: {time.time() - start_time:.2f} seconds")
    
    num_test_samples = n_test
    sample_indices = np.random.choice(len(cached_dataset), num_test_samples, replace=False)
    all_passed = True
    for i in sample_indices:
        cached_data = cached_dataset[i]
        uncached_data = uncached_dataset[i]
        
        for key in cached_data.keys():
            if not torch.allclose(cached_data[key], uncached_data[key], atol=1e-2):
                print(f"Test {i+1} failed")
                print(f"{key} not equal for test {i+1} - (index {i})")
                print(f"Shapes: {cached_data['images'].shape}, {uncached_data['images'].shape}")
                print(f"Max: {cached_data['images'].max()}, {uncached_data['images'].max()}")
                print(f"Min: {cached_data['images'].min()}, {uncached_data['images'].min()}")
                print(f"Mean: {cached_data['images'].mean()}, {uncached_data['images'].mean()}")
                print(f"Std: {cached_data['images'].std()}, {uncached_data['images'].std()}")
                
                if key == "images" and compare_diff_images:
                    # Save both images for comparison
                    cached_img = cached_data[key].permute(1, 2, 0).numpy().clip(0,1)
                    uncached_img = uncached_data[key].permute(1, 2, 0).numpy().clip(0,1)
                    plt.imsave(f"cached_img_{i+1}.jpg", cached_img)
                    plt.imsave(f"uncached_img_{i+1}.jpg", uncached_img)
                all_passed = False
                break
        else:
            print(f"Test {i+1} passed")
    
    if all_passed:
        print(f"All {num_test_samples} tests passed, datasets are same!")
    
    cached_loader = DataLoader(cached_dataset, batch_size=batch_size, shuffle=True)
    start_time = time.time()
    avg_cached_time = 0
    for i, data in enumerate(cached_loader):
        print([data[k].shape for k in data])
        time_taken = time.time() - start_time
        print(f"Time taken to load cached batch {i+1}: {time_taken:.2f} seconds")
        avg_cached_time += time_taken        
        if i == n_test - 1:
            break        
        start_time = time.time()
    print(f"\nAverage time taken to load a cached batch: {avg_cached_time / n_test:.2f} seconds\n")
    
    uncached_loader = DataLoader(uncached_dataset, batch_size=batch_size, shuffle=True)
    start_time = time.time()
    avg_time = 0
    for i, data in enumerate(uncached_loader):
        print([data[k].shape for k in data])
        time_taken = time.time() - start_time
        print(f"Time taken to load batch {i+1}: {time_taken:.2f} seconds")
        avg_time += time_taken
        if i == n_test - 1:
            break
        start_time = time.time()
    print(f"\nAverage time taken to load a batch: {avg_time / n_test:.2f} seconds")

    print(f"\nAverage speedup by caching: {avg_time / avg_cached_time:.2f}x")        


def test_dataloader():
    batch_size = 4
    
    start_time = time.time()
    train_loader, _ = create_dataloaders(root_dir=dataset_dir, batch_size=batch_size)
    print(f"Time taken to create dataloader (with batch size {batch_size}): {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    for i, data in enumerate(train_loader):
        print([data[k].shape for k in data])
        print(f"\nTime taken to load batch {i+1}: {time.time() - start_time:.2f} seconds")

        plt.imsave(f"image_{i+1}.jpg", data["images"][0].permute(1, 2, 0).numpy().clip(0, 1))

        mel = data["audios"][0].numpy()
        plt.imsave(f"mel_{i+1}.jpg", mel, cmap="viridis")
        # print(f"Min, Max for mel: {mel.min(), mel.max()}")
    
        # if "audios_raw" in data:
        #     torchaudio.save(f"audio_{i+1}.wav", data["audios_raw"][0].unsqueeze(0), 96000)
                
        if i == 1:
            break
        
        start_time = time.time()


def check_if_entire_dataset_cached():
    dataset = GreatestHitsDataset(root_dir=dataset_dir, use_cached=True)
    uncached = {"frame": [], "mel": []}
    for frame_timestamp, frame_info in tqdm(dataset.times_info.items(), desc="Checking cached dataset"):
        for frame in frame_info:
            mel_path = f"{dataset.dataset_cache_dir}/{frame_timestamp}_{frame[0]}_mel.npy"
            frame_path = f"{dataset.dataset_cache_dir}/{frame_timestamp}_{frame[0]}_frame.jpg"
            if not os.path.exists(mel_path):
                print(f"Mel not cached for {frame_timestamp}_{frame[0]}")
                uncached["mel"].append((frame_timestamp, frame[0]))
            if not os.path.exists(frame_path):
                print(f"Frame not cached for {frame_timestamp}_{frame[0]}")
                uncached["frame"].append((frame_timestamp, frame[0]))
    
    if len(uncached["mel"]) == 0 and len(uncached["frame"]) == 0:
        print("Entire cached dataset is available!")
    else:
        print(f"Mel uncached: {len(uncached['mel'])}, Frame uncached: {len(uncached['frame'])}")
        print(f"Mel uncached: {uncached['mel']}, \nFrame uncached: {uncached['frame']}")


def test_all_batch_sizes_shape():
    _, val_loader = create_dataloaders(root_dir=dataset_dir, batch_size=4, use_cached=True)
    for i, data in tqdm(enumerate(val_loader)):
        if i < len(val_loader) - 5:
            continue
        print(f"Batch {i+1} shapes:")
        print([data[k].shape for k in data])


if __name__ == '__main__':
    dataset_dir = "./vis-data-256"

    # test_cached_vs_uncached_datasets()
    check_if_entire_dataset_cached()
    # test_all_batch_sizes_shape()