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

class clip_alignment:
    
    def __init__(self, accelerator):
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        #self.image_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        print("Clip Utils Initialized")
        self.accelerator = accelerator
        
        
    
    
    def similarity_score(self, eeg_encoding, text_encoding):
        # Assuming all pairs are similar, we create a target tensor of ones
        target = torch.ones(eeg_encoding.size(0)).to(self.accelerator.device)
        
        # Calculate the loss
        loss = self.cosine_loss(eeg_encoding, text_encoding, target)
        
        return loss
        
        
        
          