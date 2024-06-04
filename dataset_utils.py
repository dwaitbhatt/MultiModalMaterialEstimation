import whisper
import glob
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, ToPILImage, Normalize
import cv2
import time
import matplotlib.pyplot as plt


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
    def __init__(self, root_dir="./vis-data-256", audio_length=5):
        self.root_dir = root_dir
        self.transform = _transform(224)
        # self.transform_to_tensor = T.ToTensor()
        self.audio_rate = 96000
        self.audio_length = audio_length
    
        self.times_info = {}
        for file in sorted(glob.glob(root_dir + "/*_times.txt")):
            with open(file, "r") as f:
                data = f.readlines()
                data = [line.strip().split() for line in data]
                data = [(float(line[0]), line[1]) for line in data if line[1] != "None"]
                self.times_info[file.split("/")[-1].split("_")[0]] = data
                
        self.all_material_names = set()
        for _, frame_info in self.times_info.items():
            self.all_material_names.update([frame[1] for frame in frame_info])

        self.all_material_names = sorted(list(self.all_material_names))
        self.material_names_dict = {name: i for i, name in enumerate(self.all_material_names)}
    
    def __len__(self):
        total_len = 0
        for key in self.times_info.keys():
            total_len += len(self.times_info[key])
        return total_len
    
    def __getitem__(self, idx):
        for key in self.times_info.keys():
            if idx < len(self.times_info[key]):
                break
            else:
                idx -= len(self.times_info[key])
        
        date_time = key
        # frames, audio, metadata = torchvision.io.read_video(f"./vis-data-256/{date_time}_denoised.mp4")
        cap = cv2.VideoCapture(self.root_dir + f"/{date_time}_denoised.mp4")
        audio = whisper.load_audio(self.root_dir + f"/{date_time}_denoised.wav", self.audio_rate)
        
        frames_info = self.times_info[date_time]
        frame_timestamp = frames_info[idx][0]
        material_name = frames_info[idx][1]

        # Whisper needs 5 seconds of audio (default for pad_or_trim is 480k samples, i.e., 5 seconds at 96k samples/sec)
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
        return {"images": frame, "audios": mel, "audios_raw":audio, "materials": material_name}
    

# Create DataLoader
def create_dataloader(root_dir="./vis-data-256", batch_size=4):
    dataset = GreatestHitsDataset(root_dir)
    print(f"\nDataset size: {len(dataset)}\n")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ == '__main__':
    batch_size = 4
    
    start_time = time.time()
    dataloader = create_dataloader(root_dir="/home/GreatestHits/vis-data-256", batch_size=batch_size)
    print(f"Time taken to create dataloader (with batch size {batch_size}): {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    for i, data in enumerate(dataloader):
        print(data["images"].shape, data["audios"].shape, data["audios_raw"].shape, data["materials"])
        print(f"\nTime taken to load batch {i+1}: {time.time() - start_time:.2f} seconds")

        plt.imsave(f"image_{i+1}.jpg", data["images"][0].permute(1, 2, 0).numpy().clip(0, 1))
        torchaudio.save(f"audio_{i+1}.wav", data["audios_raw"][0].unsqueeze(0), 96000)
        mel = data["audios"][0].numpy()

        plt.imsave(f"mel_{i+1}.jpg", mel, cmap="viridis")
        # print(f"Min, Max for mel: {mel.min(), mel.max()}")
                
        if i == 1:
            break
        
        start_time = time.time()