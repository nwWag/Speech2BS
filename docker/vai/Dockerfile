FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install build-essential -y
RUN pip install --upgrade pip && \
    pip install wandb opencv-python transformers moviepy accelerate
RUN pip install 'accelerate>=0.26.0'

COPY speech2bs/ /app/speech2bs/
COPY task.py /app/task.py
COPY default_config.yaml /app/default_config.yaml

ENTRYPOINT ["accelerate", "launch", "--config_file", "/app/default_config.yaml", \
    "task.py"]