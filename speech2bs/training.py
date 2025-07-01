from turtle import rt
from transformers import Trainer, TrainingArguments, TrainerCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import wandb


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        batch_size = z1.size(0)
        sim = torch.matmul(z1, z2.T) / self.temperature

        labels = torch.eye(batch_size, device=z1.device)
        
        loss = F.binary_cross_entropy_with_logits(sim, labels)
        return loss

    
class Speech2BsTrainer(Trainer):
    def __init__(self, contrastive, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_loss = NTXentLoss(temperature=0.5) if contrastive else None
        self.trained_steps = 0
        self.logging_steps = kwargs["args"].logging_steps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        if len(inputs) == 4:
            outputs_audio, outputs_video, outputs_text, \
                audio_embeds, video_embeds, text_embeds = model(inputs[:3], return_embeddings=True)

        loss = F.mse_loss(outputs_audio, inputs[-1], reduction='mean')
        loss += F.mse_loss(outputs_video, inputs[-1], reduction='mean') if outputs_video is not None else 0
        loss += F.mse_loss(outputs_text, inputs[-1], reduction='mean') if outputs_text is not None else 0
        loss_dict = {"recon_loss": loss}

        if self.contrastive_loss is not None:
            contrastive_loss = self.contrastive_loss(audio_embeds, video_embeds) if outputs_video is not None else 0
            contrastive_loss += self.contrastive_loss(audio_embeds, text_embeds) if outputs_text is not None else 0
            loss_dict["contrastive_loss"] = contrastive_loss
            loss += contrastive_loss

        is_training = self.model.training
        if is_training:
            if self.trained_steps % self.logging_steps == (self.logging_steps - 1): 
                wandb.log(loss_dict, step=self.trained_steps+1)
            self.trained_steps += 1

        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss, _ = self.compute_loss(model, inputs, return_outputs=True)
        return (loss, None, None)
    
def get_trainer(model, train_dataset, validation_dataset, output_folder, args):
    training_args = TrainingArguments(
    output_dir=output_folder,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    push_to_hub=False,
    report_to="wandb",
    logging_steps=args.logging_steps
)
    
    trainer = Speech2BsTrainer(
        contrastive=args.contrastive,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=default_collate     
        )
    return trainer
