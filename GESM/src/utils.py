import numpy as np
import torch
from torch.cuda.amp import GradScaler 
import math 
import os
from torch.utils.data import DataLoader
import datetime
import matplotlib.pyplot as plt
@torch.no_grad()
def plot_reconstruction(model, device, dataset, output_dir, num_examples=3):
    #unwrapping the model
    
    
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
        _, pred, mask = model(sample, mask_ratio=0.8)
        sample_with_mask = sample.to('cpu').squeeze(0)[0].numpy().reshape(-1, model.patch_size)
        pred = model.unpatchify(pred).to('cpu').squeeze(0)[0].numpy()
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

def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    Generates sine-cosine positional embeddings for 1D data.

    Args:
    - embed_dim (int): The dimensionality of the embedding for each position.
    - length (int): The length of the sequence for which to generate embeddings.
    - cls_token (bool): Whether to include an additional position for the class token.

    Returns:
    - numpy.ndarray: A matrix of size [length (+1 if cls_token), embed_dim] containing the positional embeddings.
    """
    # Create a 1D grid representing the positions in the sequence
    grid_l = np.arange(length, dtype=np.float32).reshape([1, length])

    # Generate the positional embeddings from the grid
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_l)

    # If a class token is used, prepend a zero embedding
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generates sine-cosine positional embeddings given a grid of positions.

    Args:
    - embed_dim (int): The dimensionality of the embedding for each position.
    - pos (numpy.ndarray): A 1D array of positions to be encoded.

    Returns:
    - numpy.ndarray: A matrix of size [len(pos), embed_dim] containing the positional embeddings.
    """
    # Ensure the embedding dimension is even
    assert embed_dim % 2 == 0

    # Generate the scales for the sine and cosine functions
    omega = np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)  # Scaling factors for each dimension

    pos = pos.reshape(-1)  # Flatten the position array if not already flat

    # Calculate the dot product of positions and omega, for sine and cosine separately
    out = np.einsum('m,d->md', pos, omega)  # Outer product to get (M, D/2)

    # Generate sine and cosine embeddings
    emb_sin = np.sin(out)  # Sine part of the embedding
    emb_cos = np.cos(out)  # Cosine part of the embedding

    # Concatenate sine and cosine embeddings to form the final embedding
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # Final embedding (M, D)

    return emb

def get_grad_norm(parameters, norm_type=2.0):
    """
    Calculate the norm of gradients for a list of parameters.
    Args:
    - parameters (iterable): An iterable of Parameters.
    - norm_type (float): Type of norm to use. Defaults to L2 norm.
    Returns:
    - torch.Tensor: The total norm of the gradients.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]  # Convert a single Tensor to a list for consistency
    # Filter out parameters without gradients
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0)
    # Determine device from the first parameter
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        # For infinity norm, find the max abs value among all gradients
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        # For L2 norm (or other), calculate norm across all parameters
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class GradScalerWithClip:
    """
    A wrapper around `torch.cuda.amp.GradScaler` that adds gradient clipping.
    """
    def __init__(self):
        self.scaler = GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        """
        Scales the loss, computes gradients, optionally clips them, and updates model parameters.

        Args:
        - loss (Tensor): The loss tensor.
        - optimizer (Optimizer): The optimizer.
        - clip_grad (float, optional): Max norm of the gradients.
        - parameters (iterable, optional): Iterable of parameters to clip.
        - create_graph (bool): Whether to create a computational graph for second order gradients.
        - update_grad (bool): Whether to update model parameters.

        Returns:
        - torch.Tensor or None: The norm of the gradients if `clip_grad` is not None; otherwise None.
        """
        # Backward pass with scaled loss
        self.scaler.scale(loss).backward(create_graph=create_graph)

        norm = None  # Default gradient norm is None
        if update_grad:
            # If gradient clipping is enabled, unscale and clip gradients
            if clip_grad is not None and parameters is not None:
                # Unscale gradients before clipping
                self.scaler.unscale_(optimizer)
                # Clip gradients and get their norm
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self.scaler.unscale_(optimizer)
                # Calculate gradient norm without clipping
                if parameters is not None:
                    norm = get_grad_norm(parameters)

            # Step the optimizer and update the scaler
            self.scaler.step(optimizer)
            self.scaler.update()

        return norm

    def state_dict(self):
        return self.scaler.state_dict()

    def load_state_dict(self, state_dict):
        self.scaler.load_state_dict(state_dict)



def adjust_learning_rate(optimizer, current_epoch, 
                         num_epoch = 100, 
                         warmup_epochs = 40, 
                         lr = 0.001, 
                         min_lr = 0):
    
    
    """
    Adjusts the learning rate based on the current epoch, following a schedule that initially increases the learning rate linearly during a warmup phase, then decays it according to a half-cycle cosine formula.

    Args:
    - optimizer: The optimizer for which the learning rate needs to be adjusted.
    - current_epoch: The current epoch number during training.
    - config: A configuration object containing settings for the learning rate schedule, including the initial learning rate (`lr`), the minimum learning rate (`min_lr`), the total number of epochs (`num_epoch`), and the number of warmup epochs (`warmup_epochs`).

    Returns:
    - float: The adjusted learning rate.
    """
    # During the warmup phase, increase the learning rate linearly
    if current_epoch < 40:
        lr = 0.001 * current_epoch / 40
    else:
        # After warmup, apply a half-cycle cosine decay to the learning rate
        lr = 0 + (lr - min_lr) * 0.5 * (
            1. + math.cos(
                math.pi * (current_epoch - warmup_epochs) / (num_epoch - warmup_epochs)
            )
        )

    # Apply the calculated learning rate to all parameter groups in the optimizer
    for param_group in optimizer.param_groups:
        # Adjust the learning rate, considering an optional scaling factor if present
        param_group["lr"] = lr * param_group.get("lr_scale", 1.0)

    return lr


def save_training_checkpoint(current_epoch, model, optimizer, grad_scaler, save_directory):
    """
    Saves the training checkpoint to a specified directory.

    Args:
    - config: The training configuration settings used for this training session.
    - current_epoch: The current epoch number at the time of saving.
    - model: The model being trained.
    - optimizer: The optimizer used for training.
    - grad_scaler: The gradient scaler used for mixed precision training.
    - save_directory: The directory path where the checkpoint will be saved.

    The function creates the save directory if it doesn't already exist and saves a file named 'checkpoint.pth' containing the model state, optimizer state, current epoch, gradient scaler state, and the training configuration.
    """
    # Ensure the directory exists; create it if it doesn't
    os.makedirs(save_directory, exist_ok=True)

    # Prepare the data to be saved
    checkpoint_data = {
        'model': model.state_dict(),  # Model parameters
        'optimizer': optimizer.state_dict(),  # Optimizer state
        'epoch': current_epoch,  # Current epoch
        'scaler': grad_scaler.state_dict(),  # Gradient scaler state
        
    }

    # Define the path to the checkpoint file
    checkpoint_file_path = os.path.join(save_directory, f'checkpoint-ep-{current_epoch}.pth')

    # Save the checkpoint data to the file
    torch.save(checkpoint_data, checkpoint_file_path)