import os
import tqdm
import whisper
import numpy as np
import glob
import os
from tqdm.auto import tqdm
import cv2
import json


def read_timestamp_info():
    global times_info, all_material_names, mat_to_ind, ind_to_mat
    
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

    json.dump(times_info, open(root_dir + "-processed/times_info.json", "w"))


def save_processed_dataset():
    all_material_annotations = {}
    audio_rate = 96000
    
    for date_time, frame_info in tqdm(times_info.items(), total=len(times_info), desc="Saving"):
        cap = cv2.VideoCapture(root_dir + f"/{date_time}_denoised.mp4")
        audio = whisper.load_audio(root_dir + f"/{date_time}_denoised.wav", audio_rate)
                
        audio_length = 5
    
        frame_timestamps = [frame[0] for frame in frame_info]
        material_names = [frame[1] for frame in frame_info]
    
        for i, frame_timestamp in tqdm(enumerate(frame_timestamps), total=len(frame_timestamps), leave=False):
            audio_start_time = frame_timestamp - audio_length / 2
            audio_start_idx = int(audio_start_time * audio_rate)
            audio = audio[audio_start_idx : audio_start_idx + audio_rate * audio_length]
            audio = whisper.pad_or_trim(audio, audio_rate * audio_length)
            mel = whisper.log_mel_spectrogram(audio)
            np.save(root_dir + f"-processed/{date_time}_{frame_timestamp}_mel.npy", mel.numpy(), allow_pickle=True)

            cap.set(cv2.CAP_PROP_POS_MSEC, frame_timestamp * 1000)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(root_dir + f"-processed/{date_time}_{frame_timestamp}_frame.jpg", frame)
    
            all_material_annotations[f"{date_time}_{frame_timestamp}"] = material_names[i]
    
    np.save(root_dir + f"-processed/all_material_annotations.npy", all_material_annotations, allow_pickle=True)


if __name__ == "__main__":
    root_dir = "./vis-data-256"
    os.makedirs(root_dir + "-processed/", exist_ok=True)

    read_timestamp_info()
    save_processed_dataset()