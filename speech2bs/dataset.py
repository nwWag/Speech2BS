from torch.utils.data import Dataset, random_split
import torch

class Speech2BSDataset(Dataset):
    def __init__(self, video, audio, text, bs_weights, window_size=8):
        # Three modalities
        self.audio = audio
        self.video = video
        self.text = text
        
        # Check which modalities are present
        self.has_video = False
        self.has_text = False
        if video is not None: self.has_video = True
        if text is not None: self.has_text = True

        # Labels
        self.bs_weights = bs_weights
        # Moving window size
        self.window_size = window_size

        # Dataset must have the same length
        assert len(audio) == len(bs_weights), "Audio and BS weights must have the same length " + str(len(audio)) + " " + str(len(bs_weights))
        if self.has_video:
            assert len(video) == len(audio), "Video and Audio must have the same length " + str(len(video)) + " " + str(len(audio))
        if self.has_text:
            assert len(text) == len(audio), "Text and Audio must have the same length " + str(len(text)) + " " + str(len(audio))
    
    def __len__(self):
        return len(self.bs_weights)
    
    def __getitem__(self, index):
        diff = index - self.window_size
        # Ensure to stay in dataset including window
        if diff < 0:
            index -= diff
        
        # Return depending on available modalities
        if self.has_text and self.has_video:
            return self.video[index-self.window_size:index].float(), \
                   self.audio[index-self.window_size:index].float(), \
                   self.text[index-self.window_size:index].long(), \
                   self.bs_weights[index].float()
        if self.has_video:
            return self.video[index-self.window_size:index].float(), \
                   self.audio[index-self.window_size:index].float(), \
                   self.bs_weights[index].float()
        else:
            return self.audio[index-self.window_size:index].float(), \
                self.bs_weights[index].float()


def get_validation_dataset(dataset, val_size=0.2, seed=42):
    """Create a validation dataset from the given dataset with given seed"""
    generator = torch.Generator().manual_seed(seed)
    total_size = len(dataset)
    val_length = int(total_size * val_size)
    train_length = total_size - val_length
    return random_split(dataset, [train_length, val_length], generator=generator)