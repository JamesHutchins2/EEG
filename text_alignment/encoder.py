import sys
import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
import torch.nn.functional as F
import utils as ut
from transformers import CLIPTokenizer, CLIPTextModel, CLIPProcessor
from accelerate import Accelerator
import mne
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from PIL import Image
import requests
import torch.nn.functional as F

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
        self.cosine_loss = nn.CosineEmbeddingLoss()
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
        
        self.decoder = MaskedAutoDecoder()
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


    def forward_encoder(self, x):
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
        #x, mask, ids_restore = self.random_masking(x, mask_ratio)  # Mask input data

        # Append class token and its positional embedding
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Process through transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x#, mask, ids_restore

    
    
    def forward_loss(self, x, predictions, target):
        
        
        #print("in the loss function: ")
        #print(f"taget shape: {target.shape}")
        #print(f"predictions shape: {predictions.shape}")
        #print(f"text embeddings shape: {x.shape}")
        
        
        #squeeze the text embeddings, and the predictions along dim 1
        x = x.squeeze(1)
        predictions = predictions.squeeze(1)
        
        print(f"taget shape: {predictions.shape}")
        print(f"x: {x.shape}")
        
        
        loss = 1 - nn.functional.cosine_similarity(x, predictions, dim=-1).mean()

        loss_mse = F.mse_loss(predictions.float(), x.float(), reduction="mean")
        
        
        
        loss = loss + loss_mse
        
        # Calculate the loss
        #loss = self.cosine_loss(predictions, x, target)
        
        print(f"loss: {loss}")
        #print loss shape
        print(f"shape of loss: {loss.shape}")
        return loss

    def forward(self, eeg_input, text_token, target):
        # * just to be sure
        
        # ! need to replace imgs with the text encodings
        
        
        latent = self.forward_encoder(eeg_input)
        
        pred = self.decoder.forward(latent)
        #print(pred.shape)
        
        loss = self.forward_loss(pred,text_token, target) 
        
        #print(f"shape of loss: {loss.shape}")
        
        return loss, pred 


class ReduceSequenceLinearly(nn.Module):
    def __init__(self, input_dim, seq_len, output_dim):
        super().__init__()
        self.reduce = nn.Linear(seq_len, 1)  # Reducing from seq_len to 1
        self.final_linear = nn.Linear(input_dim, output_dim)  # Final adjustment to match desired output_dim

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # Change to [batch_size, input_dim, seq_len] for linear reduction
        x = self.reduce(x)  # Reduce seq_len to 1
        x = x.permute(0, 2, 1)  # Back to [batch_size, 1, input_dim]
        x = self.final_linear(x)  # Adjust dimensions if necessary
        return x



class MaskedAutoDecoder(nn.Module):
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

        self.patch_size = patch_size
        num_patches = int(time_len / patch_size)
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Initialize the decoder part of the MAE
        # It decodes the encoded features back to the original data dimensionality
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_reduction = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim),  # First, a linear layer that keeps the dimension.
            nn.LayerNorm(decoder_embed_dim),  # Optional: normalization for stability.
            ReduceSequenceLinearly(decoder_embed_dim, num_patches, in_chans * patch_size)  # Then, our custom reduction.
        )
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
        
        
        decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize the patch embedding weights similar to nn.Linear's initialization method
        """w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))"""

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

    

    def forward_decoder(self, x):
        """
        Decodes the encoded representation back into the original data space.

        Args:
        - x (Tensor): Encoded data.
        - ids_restore (Tensor): Indices to restore the original ordering of the sequence.

        Returns:
        - The decoded representation of the data.
        """
        
        
        print(f"decoder input shape: {x.shape}")
        x = self.decoder_embed(x)  # Embed decoded tokens
        
        print(f"decoder input shape after embedding: {x.shape}")
        x = x + self.decoder_pos_embed  # Add positional embeddings

        print(f"decoder input shape after adding positional embeddings: {x.shape}")
        # Process through decoder transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = x[:, 1:, :]
        x = self.decoder_reduction(x)  # Project back to the original data space
        print(f"decoder output shape: {x.shape}")
        
        
        return x

    
    
    

    def forward(self, encoder_output):
        # * just to be sure
        mask_ratio = 0
        # ! need to replace imgs with the text encodings
        print(f"encoder_output shape: {encoder_output.shape}")
        
        pred = self.forward_decoder(encoder_output)
       
        return pred
































class Encoder(nn.Module):
    def __init__(self, time_len=512, patch_size=4, embed_dim=1024, in_chans=128,
                 depth=24, num_heads=16, decoder_embed_dim=512, 
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        self.patch_embed = PatchEmbed1D(time_len, patch_size, in_chans, embed_dim)
        num_patches = int(time_len / patch_size)
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

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



class JointEmbeddingEncoderDecoder(nn.Module):
    def __init__(self, time_len=512, patch_size=4, embed_dim=1024, in_chans=128,
                 depth=24, num_heads=16, decoder_embed_dim=512, 
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

       
        num_patches = int(time_len / patch_size)
        
        
        

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

        x = self.decoder_embed(x)

        x = x + self.decoder_pos_embed[:, 1:, :]
    
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)

        return x