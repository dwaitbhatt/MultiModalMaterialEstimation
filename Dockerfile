# Base image
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# Remove existing NVIDIA repository lists to avoid GPG errors
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

# Update and install necessary packages
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git wget unzip

RUN pip install openai-whisper opencv-python transformers torchvision wandb ipdb matplotlib tqdm scikit-learn

# Set the working directory
WORKDIR /MultiModalMaterialEstimation

COPY \
  config_test.json \
  dataset_utils.py \
  preprocessing.sh \
  process_greatest_hits.py \
  train.py \
  test.py \
  setup_after_docker.sh \
  ckpt.pth \
  test_set_times_info.json \
  ./

RUN chmod +x setup_after_docker.sh

CMD ["/bin/bash", "setup_after_docker.sh"]