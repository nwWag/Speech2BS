from moviepy import VideoFileClip
import torchaudio
import numpy as np
import subprocess
import os
import torch
import cv2
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import wandb

def video2tensor(video_path, target_size=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if target_size is not None:
        frames = np.zeros((total_frames, target_size, target_size, 3), dtype=np.uint8)
    else:
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frames = np.zeros((total_frames, height, width, 3), dtype=np.uint8)
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if target_size is not None:
            frame = cv2.resize(frame, (target_size, target_size))
        frames[i] = frame
    cap.release()
    
    video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)

    return video_tensor

def video2sound(video_path, audio_path, n_frames):
    if(not os.path.exists(audio_path)):
        subprocess.run([
        'ffmpeg', '-y', '-i', video_path,
        '-vn',          
        '-acodec', 'pcm_s16le',
        '-ar', '8000',
        '-ac', '2', 
        audio_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    audio, _ = torchaudio.load(audio_path)
    audio = audio.mean(0)
    max_frames = audio.shape[0] // n_frames * n_frames
    audio = audio[:max_frames]
    
    return  audio.reshape(n_frames, -1)

def video2text(video_path, audio_path, text_path, target_size):
    def extend_chars(chars, target_size):
        prev_len = len(chars)
        factor = target_size // prev_len
        extended_chars = []
        for c in chars:
            extended_chars.extend([c for _ in range(factor)])

        while len(extended_chars) < target_size:
            i = 0
            while i < len(extended_chars):
                if extended_chars[i] == ' ':
                    extended_chars.insert(i, ' ')
                    i += 2  # Überspringe das gerade verdoppelte Leerzeichen
                else:
                    i += 1
                if len(extended_chars) >= target_size:
                    break

                    

        return extended_chars[:target_size]

    if(not os.path.exists(audio_path)):
        subprocess.run([
        'ffmpeg', '-y', '-i', video_path,
        '-vn',          
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '2', 
        audio_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    if(not os.path.exists(text_path)):
        audio, _ = torchaudio.load(audio_path)
        audio = audio.mean(0)

        model_name = "openai/whisper-base"  # openai/whisper-large-v2"
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)

        max_input_length = 25 * 16000  # 30 seconds of audio at 16kHz
        transcription = ""
        for start in range(0, len(audio), max_input_length):
            end = min(start + max_input_length, len(audio))
            chunk = audio[start:end]
            inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features

            forced_decoder_ids = processor.get_decoder_prompt_ids(language="de", task="transcribe")
            with torch.no_grad():
                predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcription += text + " "
        
        with open(text_path, "w", encoding="utf-8") as file:
            file.write(transcription)
    
    else:
        with open(text_path, "r", encoding="utf-8") as file:
            transcription = file.read()

    chars = list(transcription.strip())
    if len(chars) >= target_size:
        return chars[:target_size], transcription
    else:
        return extend_chars(chars, target_size), transcription


def csv2tensor(csv_path):
    weights = np.genfromtxt(csv_path, skip_header=2,
                            max_rows=52,  delimiter=",")[:, 1:]

    weight_names = np.genfromtxt(csv_path, skip_header=2,
                                 max_rows=52,  delimiter=",", usecols=0, dtype=str)
    
    return torch.from_numpy(weights.T), weight_names

def extract_tensors(mode, input_folder_path, output_folder_path, args):
    bs_weights, _ = csv2tensor(os.path.join(input_folder_path, "weights.csv"))
    audio = video2sound(os.path.join(input_folder_path, "video.mp4"), os.path.join(output_folder_path, "sound8.wav"), bs_weights.shape[0])
    max_frames = min(bs_weights.shape[0], audio.shape[0])

    if args.include_video:
        video = video2tensor(os.path.join(input_folder_path, "video.mp4"), args.target_size)
        max_frames = min(max_frames, video.shape[0])
        video = video[:max_frames].contiguous()
    
    if args.include_text:
        text, transcription = video2text(os.path.join(input_folder_path, "video.mp4"), 
                          os.path.join(output_folder_path, "sound16.wav"), 
                          os.path.join(output_folder_path, "sound16.txt"), 
                          max_frames)
        text = torch.tensor([ord(c) for c in text], dtype=torch.int32).contiguous()
        trans_table = wandb.Table(columns=["transcription"])
        trans_table.add_data(transcription)
        wandb.log({str(mode) + "_transcription": trans_table})

    bs_weights = bs_weights[:max_frames].contiguous()
    audio = audio[:max_frames].contiguous()

    shape_table = wandb.Table(columns=["name", "shape"])


    shape_table.add_data("bs_weights", str(bs_weights.shape))
    shape_table.add_data("audio", str(audio.shape))
    if args.include_video:
        shape_table.add_data("video", str(video.shape))
    if args.include_text:
        shape_table.add_data("text", str(text.shape))


    wandb.log({str(mode) + "_shapes": shape_table})

    return bs_weights, audio, video if args.include_video else None, text if args.include_text else None