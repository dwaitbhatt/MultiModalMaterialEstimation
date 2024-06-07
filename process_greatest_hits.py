import os
import tqdm
import torchvision.io
import torchaudio
import whisper
from PIL import Image
import numpy as np
import glob
import os
from tqdm.auto import tqdm
import librosa
import matplotlib.pyplot as plt
import cv2
import ipdb


def read_timestamp_info():
    global times_info, all_material_names, mat_to_ind, ind_to_mat
    
    root_dir = "./vis-data-256"
    times_info = {}
    for file in tqdm(sorted(glob.glob(root_dir + "/*_times.txt")), desc="Reading timestamp data"):
        with open(file, "r") as f:
            data = f.readlines()
            data = [line.strip().split() for line in data]
            data = [(float(line[0]), line[1]) for line in data if (line[1] != "None") and (line[2] == "hit")]
            times_info[file.split("/")[-1].split("_")[0]] = data
        
    all_material_names = set()
    for _, frame_info in times_info.items():
        all_material_names.update([frame[1] for frame in frame_info])

    all_material_names = sorted(list(all_material_names))
    mat_to_ind = {name: i for i, name in enumerate(all_material_names)}
    ind_to_mat = {i: name for i, name in enumerate(all_material_names)}


def save_processed_dataset():
    root_dir = "./vis-data-256"
    os.makedirs(f"./vis-data-256-processed/", exist_ok=True)
    all_material_annotations = {}
    audio_rate = 96000
    
    for date_time, frame_info in tqdm(times_info.items(), total=len(times_info), desc="Saving"):
        # Load the audio and video
        # video = torch.load(f"./vis-data-256/{date_time}_denoised.mp4")
        # frames, _, metadata = torchvision.io.read_video(root_dir + f"/{date_time}_denoised.mp4")
        # audio = audio[0]
        # video_rate = metadata["video_fps"]
        # audio_rate = metadata["audio_fps"]
        
        cap = cv2.VideoCapture(root_dir + f"/{date_time}_denoised.mp4")
        audio = whisper.load_audio(root_dir + f"/{date_time}_denoised.wav", audio_rate)
        
        
        audio_length = 5
    
        # Get the timestamps of the frames in the video
        frame_timestamps = [frame[0] for frame in frame_info]
        material_names = [frame[1] for frame in frame_info]
    
        for i, frame_timestamp in tqdm(enumerate(frame_timestamps), total=len(frame_timestamps), leave=False):
            # Save the audio for each frame
            # audio_start = frame_timestamp - 0.5
            # audio_end = frame_timestamp + 0.5
            # audio_start_idx = int(audio_start * audio_rate)
            # audio_end_idx = int(audio_end * audio_rate)
            # torch.save(audio[audio_start_idx:audio_end_idx].unsqueeze(0), f"./vis-data-256-processed/{date_time}_{frame_timestamp}_audio.pt")
            audio_start_time = frame_timestamp - audio_length / 2
            audio_start_idx = int(audio_start_time * audio_rate)
            audio = audio[audio_start_idx : audio_start_idx + audio_rate * audio_length]
            audio = whisper.pad_or_trim(audio, audio_rate * audio_length)
            mel = whisper.log_mel_spectrogram(audio)
            np.save(f"./vis-data-256-processed/{date_time}_{frame_timestamp}_mel.npy", mel.numpy(), allow_pickle=True)

            # mel_arr = mel.cpu().numpy()
            # mel_arr = 255 * (mel_arr - mel_arr.min()) / (mel_arr.max() - mel_arr.min())
            # mel_arr = mel_arr.astype(np.uint8)
            # im = Image.fromarray(mel_arr)
            # ipdb.set_trace()
            # im.save(f"./vis-data-256-processed/{date_time}_{frame_timestamp}_mel_img.png")
            # torchaudio.save(f"./vis-data-256-processed/{date_time}_{frame_timestamp}_audio.wav", np.expand_dims(audio, axis=0), audio_rate)
            # Save the video frames
            # frame_idx = int(frame_timestamp * video_rate)
            # torch.save(frames[frame_idx].permute(2, 0, 1), f"./vis-data-256-processed/{date_time}_{frame_timestamp}_frame.pt")
            # im = Image.fromarray(frames[frame_idx].numpy())
            # im.save(f"./vis-data-256-processed/{date_time}_{frame_timestamp}_frame.jpg")
    
            cap.set(cv2.CAP_PROP_POS_MSEC, frame_timestamp * 1000)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"./vis-data-256-processed/{date_time}_{frame_timestamp}_frame.jpg", frame)
    
            # Add the material name to all_material_annotations
            all_material_annotations[f"{date_time}_{frame_timestamp}"] = material_names[i]
    
    np.save("./vis-data-256-processed/all_material_annotations.npy", all_material_annotations, allow_pickle=True)


if __name__ == "__main__":
    read_timestamp_info()
    save_processed_dataset()