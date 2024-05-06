import sys
import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
import torch.nn.functional as F
import utils as ut

class PatchEmbed1D(nn.Module):
    """
    A class for converting 1D data into patch embeddings.
    
    It divides the data into patches and projects each patch into an embedding space.
    
    Parameters:
    - time_len: The length of the time series data.
    - patch_size: The size of each patch.
    - in_chans: Number of input channels (features per time point).
    - embed_dim: The dimensionality of the output embedding space.
    """
    
    def __init__(self, time_len=224, patch_size=1, in_chans=128, embed_dim=256):
        # Initialize the parent class (nn.Module)
        super().__init__()
        
        # Calculate the number of patches by dividing the total length by the patch size
        num_patches = time_len // patch_size
        
        # Initialize attributes
        self.patch_size = patch_size
        self.time_len = time_len
        self.num_patches = num_patches

        # Define a convolutional layer to project the input data into the embedding space
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Forward pass of the module.
        
        Parameters:
        - x: The input data of shape (Batch size, Channels, Length of the data)
        
        Returns:
        - The patch embeddings of the input data.
        """
        # Ensure x is in the correct shape: (Batch size, Channels, Data length)
        B, C, V = x.shape
        
        # Project the input data into the embedding space and reshape
        # The 'proj' layer outputs (Batch size, embed_dim, Num patches)
        # We transpose to make it (Batch size, Num patches, embed_dim) for further processing
        x = self.proj(x).transpose(1, 2).contiguous()
        
        return x

class MaskedAutoencoder(nn.Module):
    """
    A Masked Autoencoder for 1D data (e.g., time series), using a transformer-based architecture.
    
    This model is designed to encode 1D input data into a lower-dimensional space and then decode 
    it back to its original dimension, with a focus on reconstructing the original data from 
    partial (masked) inputs. It features a Vision Transformer (ViT) backbone for both encoding and 
    decoding processes.
    
    Parameters:
    - time_len: Length of the input time series.
    - patch_size: Size of each patch into which the input data is divided.
    - embed_dim: Dimensionality of the embedding space for the encoder.
    - in_chans: Number of input channels.
    - Various parameters for configuring the depth and heads of the transformer model, along with other hyperparameters.
    """
    
    def __init__(self, time_len=512, patch_size=4, embed_dim=1024, in_chans=128,
                 depth=24, num_heads=16, decoder_embed_dim=512, 
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # Initialize the encoder part of the MAE
        # This involves embedding the input patches and applying transformer blocks to them
        self.patch_embed = PatchEmbed1D(time_len, patch_size, in_chans, embed_dim)
        num_patches = int(time_len / patch_size)
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Initialize the decoder part of the MAE
        # It decodes the encoded features back to the original data dimensionality
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans * patch_size)

        # Store some parameters and initializations
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes the weights of the model, setting up specific initial values for different types of layers.
        This includes setting up the positional embeddings with a sin-cos pattern, initializing weights for the patch embedding,
        class token, mask token, and standard layers (Linear, LayerNorm, Conv1d) following best practices.
        """
        
        # Initialize positional embeddings with sin-cos pattern for encoder and decoder
        # This uses a utility function to generate the embeddings, assuming it creates embeddings suitable for 1D data
        pos_embed = ut.get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize the patch embedding weights similar to nn.Linear's initialization method
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize class and mask tokens with normal distribution
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # Apply custom initialization to all layers in the model
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Applies custom initialization for different types of layers within the model.
        """
        if isinstance(m, nn.Linear):
            # Initialize Linear layers with Xavier uniform distribution
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                # Set biases to zero
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # Initialize LayerNorm layers with biases set to zero and weights set to one
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            # Initialize Conv1d layers with normal distribution for weights
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                # Set biases to zero for Conv1d layers
                nn.init.constant_(m.bias, 0)

    def patchify(self, x):
        """
        Converts images into patches for processing.
        
        Parameters:
        - x: A tensor of eeg data, expected to be in shape (N, C, L), where
        N is the batch size, C is the number of channels, and L is the length of the time series.
        
        Returns:
        - A tensor of patches ready for the model to process.
        """
        p = self.patch_embed.patch_size  # The size of each patch
        assert x.ndim == 3 and x.shape[1] % p == 0  # Ensure the image dimensions are compatible with the patch size

        # Reshape the images into patches
        x = x.reshape(shape=(x.shape[0], x.shape[1] // p, -1))
        return x

    def unpatchify(self, x):
        """
        Reverts patches back to their original image format.
        
        Parameters:
        - x: A tensor of patches processed by the model.
        
        Returns:
        - A tensor of EEG data reconstructed from the patches.
        """
        p = self.patch_embed.patch_size  # The size of each patch
        h = x.shape[1]  # The height dimension, derived from the patch tensor

        # Reshape the patches back into eeg data
        x = x.reshape(shape=(x.shape[0], -1, x.shape[2] // p))
        return x.transpose(1, 2)  # Rearrange dimensions to match original image format


    def random_masking(self, x, mask_ratio):
        """
        Randomly masks parts of the input sequence based on a given mask ratio, optionally focusing more on a specific range.

        Args:
        - x (Tensor): The input data of shape [N, L, D], where N is the batch size, L is the sequence length, and D is the feature dimension.
        - mask_ratio (float): The proportion of the sequence to be masked.

        Returns:
        - x_masked (Tensor): The masked input data.
        - mask (Tensor): The binary mask indicating masked (1) and unmasked (0) positions.
        - ids_restore (Tensor): Indices to restore the original ordering of the sequence after shuffling.
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))  # Calculate the number of positions to keep unmasked
            
        noise = torch.rand(N, L, device=x.device)  # Generate random noise for shuffling
       
        # Shuffle based on noise and separate indices for keeping and removing
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        # Mask the data by keeping only the unmasked positions
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Create a binary mask for masked positions
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """
        Encodes the input data, applying positional embeddings and masking.

        Args:
        - x (Tensor): Input data.
        - mask_ratio (float): Ratio of positions in the input sequence to be masked.

        Returns:
        - The encoded representation of the input.
        - The mask used during encoding.
        - Indices for restoring original input order.
        """
        x = self.patch_embed(x)  # Embed the patches
        x = x + self.pos_embed[:, 1:, :]  # Add positional embeddings, excluding class token position
        x, mask, ids_restore = self.random_masking(x, mask_ratio)  # Mask input data

        # Append class token and its positional embedding
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Process through transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Decodes the encoded representation back into the original data space.

        Args:
        - x (Tensor): Encoded data.
        - ids_restore (Tensor): Indices to restore the original ordering of the sequence.

        Returns:
        - The decoded representation of the data.
        """
        x = self.decoder_embed(x)  # Embed decoded tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # Concatenate mask tokens, excluding class token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # Unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # Re-append class token
        x = x + self.decoder_pos_embed  # Add positional embeddings

        # Process through decoder transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)  # Project back to the original data space
        x = x[:, 1:, :]  # Remove class token for the final output

        return x
    
    def forward_loss(self, x, predictions, mask):
        """
        Calculate the Mean Squared Error (MSE) loss between the predictions and the ground truth,
        but only for the regions that were masked out in the input.
        """
        # Transpose the data to match the prediction shape [N, Time, channels] -> [N, channels, Time]
        x = x.transpose(1, 2)

        # Convert the transposed images into patches
        target = self.patchify(x)

        # Calculate MSE loss for each patch
        mse_loss_per_patch = (predictions - target) ** 2
        mean_mse_loss_per_patch = mse_loss_per_patch.mean(dim=-1)  # Average loss per patch

        # Calculate the average loss only over the masked (removed) patches
        if mask.sum() == 0:  # Check if the mask is empty to avoid division by zero
            average_loss = mean_mse_loss_per_patch.sum()
        else:
            average_loss = (mean_mse_loss_per_patch * mask).sum() / mask.sum()

        return average_loss

    def forward(self, imgs, mask_ratio):
        
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore) 
        loss = self.forward_loss(imgs, pred, mask) 
        
        return loss, pred, mask




class TextAlignedDecoder(nn.Module):
    def __init__(self, time_len=512, patch_size=4, embed_dim=1024, in_chans=128,
                 depth=24, num_heads=16, decoder_embed_dim=512, 
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        self.patch_embed = PatchEmbed1D(time_len, patch_size, in_chans, embed_dim)
        num_patches = int(time_len / patch_size)
        

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans * patch_size)

        self.patch_size = patch_size
        self.embed_dim = embed_dim


    def forward(self, x):

        x = self.patch_embed(x)

        x = x + self.pos_embed[:, 1:, :]
    
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)

        return x

