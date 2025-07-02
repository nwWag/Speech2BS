FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install build-essential
RUN pip install --upgrade pip && \
    pip install wandb opencv-python transformers moviepy accelerate
RUN pip install 'accelerate>=0.26.0'

ENTRYPOINT ["/opt/conda/bin/python", 
            "task.py", 
            "-train_folder", "/data/---YOUR_TRAIN_FOLDER---", 
            "-test_folder", "/data/---YOUR_TEST_FOLDER---", 
            "-wandb_token", "---YOUR_WANDB_TOKEN---", 
            "-target_size", "224", 
            "--include_video", 
            "--include_text", 
            "--contrastive"]