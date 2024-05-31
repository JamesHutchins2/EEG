import yaml
import torch
import numpy as np
from src.encoder import MaskedAutoencoder
from src.utils import GradScalerWithClip as NativeScaler
from src.utils import plot_reconstruction
import logging
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from src.datasets import THINGS_loader as loader
import timm.optim.optim_factory as optim_factory
from src.utils import save_training_checkpoint
import matplotlib.pyplot as plt
from src.trainer import train_one_epoch, validate_one_epoch





logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)  # Capture all logs

# Configure file handler for INFO and DEBUG logs
info_handler = logging.FileHandler(f'log_reconstruction.log')
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


configs_path = 'configs.yaml'
with open(configs_path) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)
    num_epochs = configs['training_parameters']['num_epochs']
    batch_size = configs['training_parameters']['batch_size']
    checkpoint_path = configs['checkpoint']['checkpoint_path']
    log_steps = configs['logging']['log_steps']
    
    # define encoder architecture
    encoder_embedding_dimension = configs['model_parameters']['encoder']['embedding_dimension']
    encoder_path_size = configs['model_parameters']['encoder']['patch_size']
    encoder_depth = configs['model_parameters']['encoder']['depth']
    encoder_num_heads = configs['model_parameters']['encoder']['num_heads']
    encoder_mlp_ratio = configs['model_parameters']['encoder']['mlp_ratio']
    
    
    
    # define decoder architecture
    decoder_embedding_dimension = configs['model_parameters']['decoder']['embedding_dimension']
    decoder_path_size = configs['model_parameters']['decoder']['patch_size']
    decoder_depth = configs['model_parameters']['decoder']['depth']
    decoder_num_heads = configs['model_parameters']['decoder']['num_heads']
    decoder_mlp_ratio = configs['model_parameters']['decoder']['mlp_ratio']
    
    dropout = configs['model_parameters']['dropout']
    
    # define optimizer settings
    optimizer = configs['optimizer']['optimizer']
    learning_rate = configs['optimizer']['optimizer_parameters']['lr']
    weight_decay = configs['optimizer']['optimizer_parameters']['weight_decay']
    betas = configs['optimizer']['optimizer_parameters']['betas']
    eps = configs['optimizer']['optimizer_parameters']['eps']
    min_lr = configs['optimizer']['optimizer_parameters']['min_lr']
    warmum_epochs = configs['training_parameters']['warmup_epochs']
    
    
    
    # define checkpoint settings
    save_dir = configs['checkpoint']['checkpoint_dir']
    encoder_base_save_name = configs['checkpoint']['checkpoint_filename_base']
    load_checkpoint = configs['checkpoint']['load_checkpoint']
    
    
    #misc
    local_rank = configs['misc']['local_rank']
    seed = configs['misc']['seed']
    participant_sub_groups = configs['misc']['participant_sub_groups']
    plot_dir = configs['misc']['plot_dir']
    
# now lets setup training devices
if torch.cuda.device_count() > 1:
        torch.cuda.set_device(local_rank) 
        torch.distributed.init_process_group(backend='nccl')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
torch.manual_seed(seed)
np.random.seed(seed)





# now let's load the architecture

def main():
    # Load the model
    MAE_VIT_MODEL = MaskedAutoencoder(
            time_len=512,
            patch_size=encoder_path_size,
            embed_dim=encoder_embedding_dimension,
            decoder_embed_dim=decoder_embedding_dimension,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=encoder_mlp_ratio,
        )


    if load_checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        MAE_VIT_MODEL.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Loaded checkpoint from {checkpoint_path}')
        
    MAE_VIT_MODEL.to(device)
    model = MAE_VIT_MODEL
    if torch.cuda.device_count() > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(MAE_VIT_MODEL)
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)


    param_groups = optim_factory.param_groups_weight_decay(MAE_VIT_MODEL, weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas)
    # now let's setup the scaler

    loss_scaler = NativeScaler()

    logger.info(f"Start Training the autoencoder... ")
    
    
    
    for participant_group in participant_sub_groups:
        #load the data from those paricipants
        
        combined_dataset, dataloader_train, dataloader_val = loader.load_eeg_data(participant_group, 
                                                            batch_size=batch_size)
        
        

        
        
        results = []
        cor_list = []
        val_results = []
        for ep in range(num_epochs):
            if torch.cuda.device_count() > 1:
                dataloader_train.set_epoch(ep)  # Shuffle data at every epoch for distributed training

            cor = train_one_epoch(dataloader_train, optimizer, device, ep, loss_scaler, 0.8, logger, 
                                  model=model, logg_steps =log_steps, lr=learning_rate, min_lr=min_lr, num_epoch=num_epochs, warmup_epochs=warmum_epochs)
            cor_list.append(cor)
            # Save checkpoint and plot reconstruction figures periodically
            if ep % 1 == 0 and local_rank == 0:
                save_training_checkpoint(ep, MAE_VIT_MODEL, optimizer, loss_scaler, save_dir)
                plot_reconstruction(model, device, combined_dataset, plot_dir,  MAE_VIT_MODEL)
        mean_cor, mean_loss = validate_one_epoch(dataloader_val, device, 0.8, model, logger)
            
        val_results.append(
                {
                    'epoch': ep,
                    'val_loss': mean_loss,
                    'val_correlation': mean_cor
                }
        )
            
            




if __name__ == "__main__":
    main()




