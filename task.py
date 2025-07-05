import argparse
import logging
from typing_extensions import runtime
import wandb
from huggingface_hub import login
import datetime
logging.basicConfig(level=logging.INFO)
from accelerate import Accelerator

import speech2bs.preprocess as preprocess
import speech2bs.directories as directories
import speech2bs.dataset as dataset
import speech2bs.model as model
import speech2bs.training as training

# ===============================================================================
# Arguments & Login
# ===============================================================================
parser = argparse.ArgumentParser(description="Task Args.")

# Folder
parser.add_argument("-train_folder",
                    type=str,
                    help="Path to the training data folder.")
parser.add_argument("-test_folder",
                    type=str,
                    help="Path to the test data folder.")
parser.add_argument("-output_folder",
                    type=str,
                    default="/data/output/",
                    help="Path to the output data folder.")

# Model
parser.add_argument("-window_size",
                    type=int,
                    default=8,
                    help="Window size for the model input. Default is 8.")

# Training
parser.add_argument("-seed",
                    type=int,
                    default=42,
                    help="Random seed for reproducibility.")
parser.add_argument("-epochs",
                    type=int,
                    default=100,  
                    help="Number of training epochs. Default is 2.")
parser.add_argument("-batch_size",
                    type=int,
                    default=64,
                    help="Batch size for training. Default is 16.")
parser.add_argument("-learning_rate",
                    type=float,
                    default=1e-4,
                    help="Learning rate for the optimizer. Default is 2e-5.")
parser.add_argument("-weight_decay",
                    type=float,
                    default=0.01,   
                    help="Weight decay for the optimizer. Default is 0.01.")
parser.add_argument("--include_video",
                    action='store_true',
                    help="Include video in the training data. Default is False.")
parser.add_argument("--include_text",
                    action='store_true',
                    help="Include text in the training data. Default is False.")
parser.add_argument("--contrastive",
                    action='store_true',
                    help="Use contrastive loss during training. Default is False.")
parser.add_argument("--logging_steps",
                    type=int,
                    default=5,
                    help="Number of steps between logging. Default is 5.")

# Tokens
parser.add_argument("-hf_token",
                    type=str,
                    default=None)      
parser.add_argument("-wandb_token",
                    type=str,
                    default=None)       

# Dataset
parser.add_argument('--recreate', 
                    action='store_true', 
                    help="Do not use cached")
parser.add_argument('-target_size', 
                    type=int, 
                    default=224, 
                    help="Target size for video frames. Default is 224.")



args = parser.parse_args()

if(args.hf_token): login(args.hf_token)
if(args.wandb_token): wandb.login(key=args.wandb_token)

# W&B
run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
accelerator = Accelerator()
if accelerator.is_main_process:
    wandb.init(project="speech2bs", config=args, name=run_name)


# ===============================================================================
# Define Vars
# ===============================================================================
train_folder_path, output_train_folder_path, \
    test_folder_path, output_test_folder_path = directories.create_folders(args)

# ===============================================================================
# Preprocessing
# ===============================================================================
train_bs_weights, train_audio, train_video, train_text = preprocess.extract_tensors("train", train_folder_path, output_train_folder_path, args)
test_bs_weights, test_audio, test_video, test_text = preprocess.extract_tensors("test", test_folder_path, output_test_folder_path, args)

# ===============================================================================
# Datasets
# ===============================================================================
training_set = dataset.Speech2BSDataset(train_video, train_audio, train_text, train_bs_weights, window_size=args.window_size)
training_set, validation_set = dataset.get_validation_dataset(training_set, val_size=0.2, seed=args.seed)
test_set = dataset.Speech2BSDataset(test_video, test_audio, test_text, test_bs_weights, window_size=args.window_size)


# ===============================================================================
# Model
# ===============================================================================
sp2bs_model = model.Speech2BsTrans(n_audio_features=train_audio.shape[1],
                            n_output_features=train_bs_weights.shape[1],
                            args=args)
sp2bs_model = model.make_it_qat(sp2bs_model)

# ===============================================================================
# Training
# ===============================================================================
trainer = training.get_trainer(sp2bs_model, training_set, validation_set, output_train_folder_path, args)
trainer.train()