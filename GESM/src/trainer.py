import math
import sys
import torch
import numpy as np
import time
import src.utils as ut
import datetime
import os
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import timm.optim.optim_factory as optim_factory
import matplotlib.pyplot as plt
import logging


from src.utils import GradScalerWithClip as NativeScaler
from src.utils import save_training_checkpoint
from src.dataloader.dataloader import create_dataset



def train_one_epoch(data_loader_participant,
                    data_loader_target, 
                    optimizer, 
                    device, 
                    epoch, 
                    grad_scaler,
                    mask_ratio,
                    logger, 
                    config=None, 
                    model=None, 
                    logg_steps = 20, 
                    lr=0.001, min_lr=0.0, 
                    num_epoch=500, 
                    warmup_epochs=40):
    """
    Trains the model for one epoch through all batches in the data_loader.

    Args:
    - model: The model to be trained.
    - data_loader: DataLoader providing batches of data.
    - optimizer: Optimizer used for training.
    - device: The device to run the training on.
    - epoch: Current epoch number.
    - grad_scaler: Gradient scaler for mixed precision training.
    - log_writer: Logger for training metrics (optional).
    - config: Configuration object containing training settings.
    - start_time: Timestamp marking the start of training (optional).
    - model_without_ddp: Model without Distributed Data Parallel wrapper (optional).
    
    Returns:
    - Mean correlation coefficient across all batches in the epoch.
    """
    #print("Training one epoch")
    model.train()
    total_loss, total_cor = [], []
    accum_iter = 1 # Gradient accumulation steps

    for step, batch in enumerate(data_loader_participant):
        
        sample_captions = batch['caption']
        
        #use the caption to get the target data
        
        def get_matching_target_data(sample_caption, data_loader_target):
            for batch in data_loader_target:
                caption = batch['caption']
                if caption == sample_caption:
                    return batch['eeg']
        
        target_data = get_matching_target_data(sample_captions, data_loader_target)
        
        if step % accum_iter == 0:
            # Adjust learning rate per iteration, not per epoch
            
            ut.adjust_learning_rate(optimizer, step / len(data_loader_participant) + epoch,
                                    num_epoch = num_epoch, warmup_epochs = warmup_epochs, lr = lr, min_lr = min_lr)

        samples = batch['eeg'].to(device)
        
        # Perform forward pass and loss computation
        with torch.cuda.amp.autocast(enabled=True):
            loss, pred, _ = model(samples, target_data)
        
        # Scale loss and backpropagate
        grad_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=0.8)
        loss_value = loss.item()

        """ if not math.isfinite(loss_value):
            logger.info(f"Loss is {loss_value}, stopping training. Step: {step}, Epoch: {epoch}")
            sys.exit(1)"""

        # Calculate correlation coefficient after unpatchifying predictions
        pred, samples = model.unpatchify(pred).cpu().detach(), samples.cpu().detach()
        cor = torch.mean(torch.tensor([torch.corrcoef(torch.stack([p[0], s[0]]))[0, 1] for p, s in zip(pred, samples)])).item()
        
        total_loss.append(loss_value)
       
        
        total_cor.append(cor)

        if device == torch.device('cuda:0'):
            if step % logg_steps ==0:
                logger.info(f"Step loss: {np.mean(total_loss)}, LR: {optimizer.param_groups[0]['lr']}, Correlation: {np.mean(total_cor)}")
    
    logger.info(f'[Epoch {epoch}] Average loss: {np.mean(total_loss)}')

    return np.mean(total_cor), np.mean(total_loss)


def validate_one_epoch(data_loader, device, mask_ratio, model, logger):
    """
    Validates the model for one epoch through all batches in the data_loader.

    Args:
    - model: The model to be evaluated.
    - data_loader: DataLoader providing batches of data.
    - device: The device to run the evaluation on.
    - mask_ratio: Masking ratio used during the forward pass.
    - model_without_ddp: Model without Distributed Data Parallel wrapper (optional).

    Returns:
    - Mean correlation coefficient and mean loss across all batches in the epoch.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss, total_cor = [], []

    with torch.no_grad():  # No gradients needed for validation
        for batch in data_loader:
            samples = batch['eeg'].to(device)

            # Perform forward pass only
            with torch.cuda.amp.autocast(enabled=True):
                loss, pred, _ = model(samples, mask_ratio=mask_ratio)
            
            # Unpatchify predictions for correlation computation
            pred, samples = model.unpatchify(pred).cpu().detach(), samples.cpu().detach()

            # Compute correlation coefficient
            cor = torch.mean(torch.tensor([
                torch.corrcoef(torch.stack([p[0], s[0]]))[0, 1] for p, s in zip(pred, samples)
            ])).item()

            # Compute mean squared error loss
            mse_loss = torch.nn.functional.mse_loss(pred, samples)

            total_loss.append(mse_loss.item())
            total_cor.append(cor)

    # Log and return the average loss and correlation coefficient
    logger.info(f'Validation - Average loss: {np.mean(total_loss)}, Average correlation: {np.mean(total_cor)}')
    return np.mean(total_cor), np.mean(total_loss)





"""if __name__ == '__main__':
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    config = Config()
    config.local_rank = local_rank
    main(config)
"""