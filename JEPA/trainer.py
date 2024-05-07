import math
import sys
import torch
import numpy as np
import time
import utils as ut
import datetime
import os
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import timm.optim.optim_factory as optim_factory
import matplotlib.pyplot as plt
import logging

from encoder import MaskedAutoencoder
from utils import GradScalerWithClip as NativeScaler
from utils import save_training_checkpoint
from dataset import create_dataset
from config import Config

# Set up logging
logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)  # Capture all logs

# Configure file handler for INFO and DEBUG logs
info_handler = logging.FileHandler(f'log_Mar30.log')
info_handler.setLevel(logging.INFO)
info_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
info_handler.setFormatter(info_formatter)
info_handler.addFilter(lambda record: record.levelno <= logging.INFO)  # Only logs <= INFO level

# Configure file handler for WARNING and above logs
warning_handler = logging.FileHandler('warning.log')
warning_handler.setLevel(logging.WARNING)
warning_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
warning_handler.setFormatter(warning_formatter)

# Add handlers to the logger
logger.addHandler(info_handler)
logger.addHandler(warning_handler)

def train_one_epoch(model, data_loader, optimizer, device, epoch, grad_scaler, mask_ratio, config=None, model_without_ddp=None):
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
    model.train()
    total_loss, total_cor = [], []
    accum_iter = config.accum_iter  # Gradient accumulation steps

    for step, batch in enumerate(data_loader):
        if step % accum_iter == 0:
            # Adjust learning rate per iteration, not per epoch
            ut.adjust_learning_rate(optimizer, step / len(data_loader) + epoch, config)

        samples = batch['eeg'].to(device)

        # Perform forward pass and loss computation
        with torch.cuda.amp.autocast(enabled=True):
            loss, pred, _ = model(samples, mask_ratio=mask_ratio)
        
        # Scale loss and backpropagate
        #grad_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)
        #loss_value = loss.item()

        """ if not math.isfinite(loss_value):
            logger.info(f"Loss is {loss_value}, stopping training. Step: {step}, Epoch: {epoch}")
            sys.exit(1)"""

        # Calculate correlation coefficient after unpatchifying predictions
        pred, samples = model_without_ddp.unpatchify(pred).cpu().detach(), samples.cpu().detach()
        cor = torch.mean(torch.tensor([torch.corrcoef(torch.stack([p[0], s[0]]))[0, 1] for p, s in zip(pred, samples)])).item()
        mse_loss = torch.nn.functional.mse_loss(pred, samples)
        total_loss.append(mse_loss)
        #total_loss.append(loss_value)
        total_cor.append(cor)

        if device == torch.device('cuda:0'):
            if step % config.log_steps ==0:
                logger.info(f"Step loss: {np.mean(total_loss)}, LR: {optimizer.param_groups[0]['lr']}, Correlation: {np.mean(total_cor)}")
    
    logger.info(f'[Epoch {epoch}] Average loss: {np.mean(total_loss)}')

    return np.mean(total_cor), np.mean(total_loss)

@torch.no_grad()
def main(config):

    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank) 
        torch.distributed.init_process_group(backend='nccl')

    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Dataset and DataLoader setup
    dataset_pretrain = create_dataset(raw_file_path = "/mnt/a/MainFolder/Neural Nirvana/Data/sub-02/eeg/sub-02_task-rsvp_eeg.vhdr", event_description = 'Event/E  1', batch_size=config.batch_size)
    sampler = DistributedSampler(dataset_pretrain, rank=config.local_rank) if torch.cuda.device_count() > 1 else None 
    dataloader_eeg = DataLoader(dataset_pretrain, batch_size=config.batch_size, sampler=sampler, shuffle=sampler is None, pin_memory=True)

    
    
    # encoder, and decoder unmasked data, calculating loss to original image
    # also use the encoding as the target for the context model prediction 
    target_model_encoder = MaskedAutoencoder(
        time_len=dataset_pretrain.data_len,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        decoder_embed_dim=config.decoder_embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio,
    )
    
    
    
    
    
    #takes in a masked image, and predicts some target block of it. 
    context_model = MaskedAutoencoder(
        time_len=dataset_pretrain.data_len,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        decoder_embed_dim=config.decoder_embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio,
    )
    
    #load the checkpoint weights 
    model.load_state_dict(torch.load('/mnt/a/MainFolder/Neural Nirvana/encoder_transformer/model_copy/EEG/auto_encoder/dd_enc_2.pth')['model'])

    model.to(device)

    model_without_ddp = model

    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank)

    param_groups = optim_factory.param_groups_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))

    loss_scaler = NativeScaler()

    logger.info(f"Start Training the autoencoder... Dataset size: {len(dataset_pretrain)}, Time len: {dataset_pretrain.data_len}")

    
    number_of_test_windows = 5
    m_ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.675, 0.7, 0.725, 0.75, 0.8, 1]
    results = []
    for ep in range(number_of_test_windows):
        for m in m_ratios:
            cor, mse = train_one_epoch(model, dataloader_eeg, optimizer, device, ep, loss_scaler,m, config, model_without_ddp)
            result = {'epoch': ep, 'mask_ratio': m, 'correlation': cor, 'mse': mse}
            results.append(result)
            
    return results
    
    """
    cor_list = []
    for ep in range(config.num_epoch):
        if torch.cuda.device_count() > 1:
            sampler.set_epoch(ep)  # Shuffle data at every epoch for distributed training

        cor = train_one_epoch(model, dataloader_eeg, optimizer, device, ep, loss_scaler, config, model_without_ddp)
        cor_list.append(cor)

        # Save checkpoint and plot reconstruction figures periodically
        if ep % 20 == 0 and config.local_rank == 0:
            save_training_checkpoint(config, ep, model_without_ddp, optimizer, loss_scaler, config.save_dir)
            plot_reconstruction(model, device, dataset_pretrain, config.plot_dir, config, model_without_ddp)
"""

@torch.no_grad()
def plot_reconstruction(model, device, dataset, output_dir, config, model_unwrapped, num_examples=3):
    """
    Generates and saves comparison plots of ground-truth, masked ground-truth, and reconstructed EEG data,
    but with a more compact figure size.

    Args:
    - model: The trained model used for data reconstruction.
    - device: The device on which the model is running (CPU or GPU).
    - dataset: The dataset from which samples are taken to generate plots.
    - output_dir: Directory where the plots will be saved.
    - num_examples: Number of figures to generate.
    - config: Configuration object containing model settings like mask_ratio.
    - logger: Optional logger for saving images in a log.
    - model_unwrapped: The original model without any wrappers like DataParallel.
    """
    # Make sure the ouput_dir exists
    os.makedirs(output_dir, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()
    fig, axs = plt.subplots(num_examples, 3, figsize=(15,7.5))
    axs[0,0].set_title('Ground-truth')
    axs[0,1].set_title('Masked Ground-truth')
    axs[0,2].set_title('Reconstruction')

    for ax in axs:
        sample = next(iter(dataloader))['eeg']
        sample = sample.to(device)
        _, pred, mask = model(sample, mask_ratio=config.mask_ratio)
        sample_with_mask = sample.to('cpu').squeeze(0)[0].numpy().reshape(-1, model_unwrapped.patch_size)
        pred = model_unwrapped.unpatchify(pred).to('cpu').squeeze(0)[0].numpy()
        sample = sample.to('cpu').squeeze(0)[0].numpy()
        mask = mask.to('cpu').numpy().reshape(-1)

        cor = np.corrcoef([pred, sample])[0,1]
        x_axis = np.arange(0, sample.shape[-1])
        ax[0].plot(x_axis, sample)
        s = 0
        for x, m in zip(sample_with_mask,mask):
            if m == 0:
                ax[1].plot(x_axis[s:s+len(x)], x, color='#1f77b4')
            s += len(x)
        ax[2].plot(x_axis, pred)
        ax[2].set_ylabel('cor: %.4f'%cor, weight = 'bold')
        ax[2].yaxis.set_label_position("right")

    fig_name = 'reconst-%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    fig.savefig(os.path.join(output_dir, f'{fig_name}.png'))
    plt.close(fig)


if __name__ == '__main__':
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    config = Config()
    config.local_rank = local_rank
    main(config)
