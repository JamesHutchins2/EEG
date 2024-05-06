import math
import sys
import torch
import numpy as np
import time
from dl_v_class import load_and_preprocess_eeg_data
import utils as ut
import datetime
from torch import cuda
import os
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import timm.optim.optim_factory as optim_factory
import matplotlib.pyplot as plt
import logging
from transformers import CLIPTokenizer, CLIPTextModel, CLIPProcessor
from encoder_v2 import MaskedAutoencoder
from utils import GradScalerWithClip as NativeScaler
from utils import save_training_checkpoint
from dataset_v2 import create_dataset
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

def train_one_epoch(model, data_loader, optimizer, device, epoch, grad_scaler, config=None, model_without_ddp=None):
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    #self.image_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    print("Clip Utils Initialized")
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
        text_encodings = batch['caption']
        
        if text_encodings is None:
            continue
        
        encoded_descriptions = []
        # covert to tensor
        tokenized_inputs = [processor(text=desc, return_tensors="pt", padding=True, truncation=True) for desc in text_encodings]
        for inputs in tokenized_inputs:
            # Move inputs to the correct device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            # Encode using the text encoder
            encoded_description = text_encoder(**inputs)
            encoded_descriptions.append(encoded_description)
        
        features = []
        #covert encoded_descriptions to a tesnro
        for td in encoded_descriptions:
            if hasattr(td, 'pooler_output') and td.pooler_output is not None:
                features.append(td.pooler_output)
            else:
                # Taking the mean of the last_hidden_state as a simple way to get a fixed-size vector
                # You can choose a different method based on your needs
                features.append(td.last_hidden_state.mean(dim=1))
        text_features_tensor = torch.stack(features)
        text_features_tensor = text_features_tensor.to(device)
        
        #define batch sized tensor of zeros for cosine targets
        
        targets = torch.ones(samples.size(0)).to(device) 
        
        # Perform forward pass and loss computation
        with torch.cuda.amp.autocast(enabled=True):
            loss, pred  = model(samples, text_features_tensor, targets)
        
        # Scale loss and backpropagate
        grad_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)
        loss_value = loss.item()
        print("loss value: ", loss_value)
        if not math.isfinite(loss_value):
            logger.info(f"Loss is {loss_value}, stopping training. Step: {step}, Epoch: {epoch}")
            sys.exit(1)

        # Calculate correlation coefficient after unpatchifying predictions
        #pred, samples = model_without_ddp.unpatchify(pred).cpu().detach(), samples.cpu().detach()
        #cor = torch.mean(torch.tensor([torch.corrcoef(torch.stack([p[0], s[0]]))[0, 1] for p, s in zip(pred, samples)])).item()

        total_loss.append(loss_value)
        #total_cor.append(cor)

        if device == torch.device('cuda:0'):
            if step % config.log_steps ==0:
                logger.info(f"Step loss: {np.mean(total_loss)}, LR: {optimizer.param_groups[0]['lr']}, Correlation: {np.mean(total_cor)}")
        print("end of train one epoch")
    logger.info(f'[Epoch {epoch}] Average loss: {np.mean(total_loss)}')

    return np.mean(total_loss)



def validate_one_epoch(model, data_loader, device, epoch, config=None, model_without_ddp=None, dataset_length=0):
    print("start of validate one epoch")
    """
    Validates the model for one epoch through all batches in the data_loader.
    
    Args:
    - model: The model to be validated.
    - data_loader: DataLoader providing batches of data for validation.
    - device: The device to run the validation on.
    - epoch: Current epoch number.
    - config: Configuration object containing validation settings (optional).
    - model_without_ddp: Model without Distributed Data Parallel wrapper (optional).
    
    Returns:
    - Mean loss and mean correlation coefficient across all batches in the epoch.
    """
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    
    model.eval()  # Set the model to evaluation mode
    total_loss, total_cor = [], []
    
    #get the length of the data loader
    length = dataset_length
    length = length * 0.1
    
    print(f"length of validation: {length}")
    
    
    with torch.no_grad():  # No need to track gradients during validation
        for step, batch in enumerate(data_loader):
            length = length - 1
            if length < 0:
                break
            samples = batch['eeg'].to(device)
            text_encodings = batch['caption']
            
            # The following encoding steps should be similar to the training loop
            encoded_descriptions = []
            tokenized_inputs = [processor(desc, return_tensors="pt", padding=True, truncation=True) for desc in text_encodings]
            for inputs in tokenized_inputs:
                inputs = {key: value.to(device) for key, value in inputs.items()}
                encoded_description = text_encoder(**inputs)
                encoded_descriptions.append(encoded_description)
            
            features = []
            for td in encoded_descriptions:
                if hasattr(td, 'pooler_output') and td.pooler_output is not None:
                    features.append(td.pooler_output)
                else:
                    features.append(td.last_hidden_state.mean(dim=1))
            text_features_tensor = torch.stack(features).to(device)
            
            # Targets during validation could be different based on the task
            targets = torch.ones(samples.size(0)).to(device)
            
            # Perform validation pass
            loss, pred = model(samples, text_features_tensor, targets)
            loss_value = loss.item()
            total_loss.append(loss_value)
            
            # Correlation calculation could be similar to the training loop, if applicable
            #total_cor.append(cor)

            # Logging or additional steps could be added here
            
    print(f'[Validation Epoch {epoch}] Average loss: {np.mean(total_loss)}')
    return np.mean(total_loss), np.mean(total_cor) if total_cor else 0





def main(config):

    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank) 
        torch.distributed.init_process_group(backend='nccl')

    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    
            
        
    
    
    this_time_len = 512
    
    
    model = MaskedAutoencoder(
        time_len= config.time_len,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        decoder_embed_dim=config.decoder_embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio,
    )
    #model.load_state_dict(torch.load("/mnt/a/MainFolder/Neural Nirvana/encoder_transformer/model_copy/dd_enc.pth"))
    def disable_grad_for_substrings(model, substrings):
        for name, param in model.named_parameters():
            if any(substring in name for substring in substrings):
                param.requires_grad = False
    substrings_to_freeze = ["decoder", "mask_token", "cls_token"]
    disable_grad_for_substrings(model, substrings_to_freeze)
    
    
    # load the existing model weights for the encoder
    #/mnt/a/MainFolder/Neural Nirvana/encoder_transformer/model_copy/dd_enc.pth
    #/home/hutchins/projects/def-yalda/hutchins/dd_enc_2.pth #/Neural Nirvana/encoder_transformer/model_copy
    #model.load_state_dict(torch.load("/home/hutchins/projects/def-yalda/hutchins/alignment/checkpoints/checkpoint.pth")["model"], strict=False)
    model.load_state_dict(torch.load("/mnt/a/MainFolder/Neural Nirvana/encoder_transformer/model_copy/checkpoints/THINGS_PRE_TRAINED.pth")["model"], strict=False)

    model.to(device)

    model_without_ddp = model
    
    
   

    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank)

    #param_groups = optim_factory.param_groups_weight_decay(model, config.weight_decay)
    # First, freeze all parameters
    #freeze_all_parameters(model)

    # Then, unfreeze only the decoder parameters
    #unfreeze_decoder_parameters(model)

    # Now, setup your new optimizer with the unfrozen parameters
    param_groups = optim_factory.param_groups_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))

    loss_scaler = NativeScaler()

    #logger.info(f"Start Training the autoencoder... Dataset size: {len(dataset_pretrain)}, Time len: {dataset_pretrain.data_len}")
    
    cor_list = []
    dataloaders_train = []
    
    
    datasets = []
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
    for i in range(1,7):
        dataset = load_and_preprocess_eeg_data("./data/eeg2.pth", i)
        datasets.append(dataset)

    # Combine all datasets into one
    combined_dataset = ConcatDataset(datasets)

    # Check if multiple GPUs are available for distributed training
    sampler_train = DistributedSampler(combined_dataset, rank=config.local_rank) if cuda.device_count() > 1 else None

    # Create a DataLoader for the combined dataset
    dataloader_train = DataLoader(combined_dataset, batch_size=config.batch_size, sampler=sampler_train, shuffle=sampler_train is None, pin_memory=True, collate_fn=collate_fn)

    #for dataloader in dataloaders_train:
    
    for ep in range(config.num_epoch):  # Subtract 1 so we have data for validation from the next path
            # Load the current epoch's training data
            
            
            # Training loop
            #for i in range(config.repeat_count):
            cor = train_one_epoch(model, dataloader_train, optimizer, device, ep, loss_scaler, config, model_without_ddp)
            cor_list.append(cor)
            # Validation step
            #validate_loss, validate_cor = validate_one_epoch(model, dataloader_val, device, ep, config, model_without_ddp, len(dataset_next))
            #logger.info(f"[Validation] Epoch: {ep}, Loss: {validate_loss}, Correlation: {validate_cor}")
            
            
            #print(f"[Validation] Epoch: {ep}, Loss: {validate_loss}, Correlation: {validate_cor}")

            # Your checkpointing and plotting code...
            if ep % 20 == 0 and config.local_rank == 0:
                save_training_checkpoint(config, ep, model_without_ddp, optimizer, loss_scaler, config.save_dir)
                #plot_reconstruction(model, device, dataset_pretrain, config.plot_dir, config, model_without_ddp)

    # Additional code to handle the last epoch since it won't have a "next" dataset for validation
    # You might decide to only train, or repeat the validation with the last used validation set, or any other policy fitting your project requirements





if __name__ == '__main__':
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    config = Config()
    config.local_rank = local_rank
    main(config)




"""try:
            print("trying to load training data")
            dataset_train = create_dataset(raw_file_path=paths_EEG[ep], event_file_path=paths_csv[ep], event_description='Event/E  1', batch_size=config.batch_size)
            print(f"training data for epoch {ep} loaded")
            sampler_train = DistributedSampler(dataset_train, rank=config.local_rank) if torch.cuda.device_count() > 1 else None
            print(f"training data for epoch {ep} sampled")
            dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, sampler=sampler_train, shuffle=sampler_train is None, pin_memory=True)
            print(f"training data for epoch {ep} loaded")
        except:
            # Skip bad data
            print(f"bad data for epoch {ep}")
            continue
        
        if ep == 0:
            logger.info(f"Start Training the autoencoder... Dataset size: {len(dataset_train)}, Time len: {dataset_train.data_len}")
        
        # Load the next epoch's data for validation (10% of it)
        try:
            dataset_next = create_dataset(raw_file_path=paths_EEG[ep + 1], event_file_path=paths_csv[ep + 1], event_description='Event/E  1', batch_size=config.batch_size)
            sampler_val = DistributedSampler(dataset_next, rank=config.local_rank) if torch.cuda.device_count() > 1 else None
            dataloader_val = DataLoader(dataset_next, batch_size=config.batch_size, sampler=sampler_val, shuffle=False, pin_memory=True)
        except:
            # get the next, next epoch
            try:
                dataset_next = create_dataset(raw_file_path=paths_EEG[ep + 2], event_file_path=paths_csv[ep + 2], event_description='Event/E  1', batch_size=config.batch_size)
                sampler_val = DistributedSampler(dataset_next, rank=config.local_rank) if torch.cuda.device_count() > 1 else None
                dataloader_val = DataLoader(dataset_next, batch_size=config.batch_size, sampler=sampler_val, shuffle=False, pin_memory=True)
            except:
                # get the next, next, next epoch
                try:
                    dataset_next = create_dataset(raw_file_path=paths_EEG[ep + 3], event_file_path=paths_csv[ep + 3], event_description='Event/E  1', batch_size=config.batch_size)
                    sampler_val = DistributedSampler(dataset_next, rank=config.local_rank) if torch.cuda.device_count() > 1 else None
                    dataloader_val = DataLoader(dataset_next, batch_size=config.batch_size, sampler=sampler_val, shuffle=False, pin_memory=True)
                except:
                    continue"""