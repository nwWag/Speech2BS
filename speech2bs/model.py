import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq
from transformers import GPT2Config, GPT2Model, VivitConfig, VivitModel

class Speech2BsTrans(nn.Module):
    """Overall model for Speech2Bs, including audio, video and text transformers. Seperate linear regressors for each modality."""
    def __init__(self, n_audio_features, n_output_features, args):
        super().__init__()
        
        # ===============================================================================
        # Audio Transformer
        # ===============================================================================
        # Default gpt2 transformer and linear tokenzier for audio features
        config = GPT2Config(
            vocab_size=2,
            n_positions=args.window_size,
            n_ctx=args.window_size,
            n_embd=64,
            n_layer=2,
            n_head=2
        )
        self.audio_transformer = GPT2Model(config)
        self.tokenizer = nn.Sequential(nn.Linear(n_audio_features, 64),
                                       nn.Sigmoid())
        self.audio_regressor = nn.Sequential(nn.Linear(self.audio_transformer.config.hidden_size, n_output_features),
                                       nn.Sigmoid())
       
        # ===============================================================================
        # Video Transformer
        # ===============================================================================       
        # Default vivit transformer and linear tokenzier for video features to match audio features                         
        self.video_transformer = None
        self.video_hidden_size = 0
        if args.include_video:
            self.video_hidden_size = self.audio_transformer.config.hidden_size
            config = VivitConfig(
                image_size=args.target_size,
                num_frames=args.window_size,
                num_hidden_layers=2,
                num_attention_heads=8,
                hidden_size=self.video_hidden_size
            )
            self.video_transformer = VivitModel(config)
            self.video_regressor = nn.Sequential(nn.Linear(self.video_hidden_size, n_output_features),
                                        nn.Sigmoid())

        # ===============================================================================
        # Text Transformer
        # ===============================================================================
        # Default gpt2 transformer and one hot tokenzier for unicode characters
        self.text_transformer = None
        if args.include_text:
            config = GPT2Config(
                vocab_size=256, 
                n_positions=args.window_size,
                n_ctx=args.window_size,
                n_embd=256,
                n_layer=2,
                n_head=2
            )
            self.text_tokenizer = lambda x: F.one_hot(x, num_classes=256)
            self.text_transformer = GPT2Model(config)
            self.text_down_embed = nn.Linear(256, self.audio_transformer.config.hidden_size)
            self.text_regressor = nn.Sequential(nn.Linear(self.audio_transformer.config.hidden_size, n_output_features),
                                        nn.Sigmoid())
            
    def forward(self, x, return_embeddings=False):
        # Check input modalities
        if len(x) == 2:
            video = x[0]
            audio = x[1]
        elif len(x) == 3:
            video = x[0]
            audio = x[1]
            text = x[2]
        else:
            audio = x


        # Audio
        audio_embeds = self.tokenizer(audio)
        audio_embeds = self.audio_transformer(
            inputs_embeds=audio_embeds
        ).last_hidden_state.mean(dim=1)
        bs_audio = self.audio_regressor(audio_embeds)

        # Video
        bs_video, video_embeds = None, None
        if self.video_transformer is not None:
            video_embeds = self.video_transformer(video).pooler_output
            bs_video = self.video_regressor(video_embeds)

        # Text
        bs_text, text_embeds = None, None
        if self.text_transformer is not None:
            text_embeds = self.text_tokenizer(text)
            text_embeds[text_embeds < 0] = 0
            text_embeds[text_embeds >= 255] = 0
            text_embeds = self.text_transformer(
                inputs_embeds=text_embeds
            ).last_hidden_state.mean(dim=1)
            text_embeds = self.text_down_embed(text_embeds)
            bs_text = self.text_regressor(text_embeds)

        # Return all modalities and set none if not present
        if not return_embeddings:
            return bs_audio, bs_video, bs_text
        else:
            return bs_audio, bs_video, bs_text, audio_embeds, video_embeds, text_embeds


def make_it_qat(module):
    module.qconfig = tq.get_default_qat_qconfig('fbgemm')
    tq.prepare_qat(module, inplace=True)
    return module