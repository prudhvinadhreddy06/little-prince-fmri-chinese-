!pip install nilearn openneuro-py textgrid

!openneuro-py download --dataset ds003643 --include "annotation/CN/lppCN_section*" --target-dir "/kaggle/working/ds003643-download/annotations"  --verify-hash

import os
import glob
import shutil
import math
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import subprocess
import re
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import gc
import time


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
BASE_DIR = "/kaggle/working/ds003643-download"
FEATURES_DIR = "/kaggle/input/cn-audio-features"
TEXT_FEATURES_DIR = "/kaggle/working/ds003643-download/annotations/annotation/CN/lppCN_word_embeddings_BERT.csv"
SUBJECTS_DIR = f"{BASE_DIR}"
OUTPUT_DIR = f"{BASE_DIR}/model_outputs"
MODEL_DIR = f"{OUTPUT_DIR}/models"
RESULTS_DIR = f"{OUTPUT_DIR}/results"

# Create output directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Global parameters
NUM_SUBJECTS = 37  # CN001 to CN037
NUM_EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 8     # Reduced from 16 to save memory
LEARNING_RATE = 1e-4

# Memory management settings
MEMORY_CLEANUP_PAUSE = 0.01  # Seconds to pause after memory cleanup

import torch
import torch.nn as nn
import torch.nn.functional as F

from textgrid import TextGrid, IntervalTier
import numpy as np
from transformers import BertModel, BertTokenizer


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, dim))

    def forward(self, x):
        # x shape: (B, T, D)
        return x + self.pos_embedding[:, :x.size(1), :]


class MultimodalEncoder(nn.Module):
    def __init__(self, audio_dim=1024, text_dim=768, hidden_dim=512, output_dim=486180, max_seq_len=512):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Projections
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Learnable positional encodings
        self.audio_pos_enc = LearnablePositionalEncoding(max_seq_len, hidden_dim)
        self.text_pos_enc = LearnablePositionalEncoding(max_seq_len, hidden_dim)

        # Cross-modal attention layers
        self.audio_to_text_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.text_to_audio_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1, batch_first=True)

        # Temporal processing (Conv1D + LayerNorm)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, audio, text, audio_mask=None, text_mask=None):
        # audio: (B, T_a, D_a)
        # text: (B, T_t, D_t)
        
        B, T_a, _ = audio.shape
        _, T_t, _ = text.shape

        audio_proj = self.audio_proj(audio)  # (B, T_a, H)
        text_proj = self.text_proj(text)    # (B, T_t, H)

        # Add positional encodings
        audio_proj = self.audio_pos_enc(audio_proj)
        text_proj = self.text_pos_enc(text_proj)

        # Cross-modal attention: audio attends to text
        audio_attn, _ = self.audio_to_text_attn(audio_proj, text_proj, text_proj,
                                                key_padding_mask=text_mask)

        # Cross-modal attention: text attends to audio
        text_attn, _ = self.text_to_audio_attn(text_proj, audio_proj, audio_proj,
                                               key_padding_mask=audio_mask)

        # Residual fusion
        audio_combined = torch.cat([audio_proj, audio_attn], dim=-1)  # (B, T_a, 2H)
        text_combined = torch.cat([text_proj, text_attn], dim=-1)     # (B, T_t, 2H)

        # Align sequence lengths (trim or pad for simplicity here)
        min_len = min(audio_combined.shape[1], text_combined.shape[1])
        audio_combined = audio_combined[:, :min_len, :]
        text_combined = text_combined[:, :min_len, :]
        combined = (audio_combined + text_combined) / 2  # (B, T, 2H)

        # Conv1d expects (B, C, T)
        combined = combined.transpose(1, 2)             # (B, 2H, T)
        temporal_out = self.temporal_conv(combined)     # (B, H, T)
        temporal_out = temporal_out.transpose(1, 2)     # (B, T, H)
        
        # Apply LayerNorm across hidden dim
        temporal_out = self.temporal_norm(temporal_out) # (B, T, H)

        # Mean pooling over time
        context = temporal_out.mean(dim=1)  # (B, H)

        # Output projection
        output = self.output(context)  # (B, output_dim)

        return output

class TextFeatureExtractor:
    def __init__(self, bert_model_name='bert-base-multilingual-cased'):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name).to(device)
        self.bert.eval()
    
    def load_textgrid(self, textgrid_path):
        """Load TextGrid file using TextGrid module"""
        tg = TextGrid.fromFile(textgrid_path)
        return tg
    
    def get_word_intervals(self, textgrid):
        """Extract word/letter intervals from TextGrid"""
        for tier in textgrid.tiers:
            if tier.name.lower() in ['words', 'letters', 'segments']:
                return tier.intervals
        return textgrid.tiers[0].intervals  # Fallback to first tier
    
    def get_bert_embedding(self, text):
        """Get BERT embedding for text segment"""
        inputs = self.tokenizer(text, return_tensors='pt', 
                              padding=True, truncation=True,
                              max_length=128).to(device)
        with torch.no_grad():
            outputs = self.bert(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Average pooling
    
    def create_text_features(self, intervals, num_frames, frame_rate=50):
        """Create frame-aligned text features"""
        text_features = torch.zeros((num_frames, 768)).to(device)
        
        for interval in intervals:
            if interval.mark:  # Only process non-empty intervals
                start_frame = int(interval.minTime * frame_rate)
                end_frame = int(interval.maxTime * frame_rate)
                end_frame = min(end_frame, num_frames)
                
                if start_frame < end_frame:  # Valid interval
                    embedding = self.get_bert_embedding(interval.mark)
                    text_features[start_frame:end_frame] = embedding
                    
        return text_features







def load_multimodal_data(feature_file, textgrid_file, bert_embeddings_path="/kaggle/working/ds003643-download/annotations/annotation/CN/lppCN_word_embeddings_BERT.csv"):
    """Load and align audio and text features"""
    # Load audio features
    audio_data = np.load(feature_file, allow_pickle=True).item()
    audio_frames = audio_data['frame_features']  # shape: (time, 1024)
    num_frames = audio_frames.shape[0]
    
    # Initialize text feature extractor
    text_extractor = TextFeatureExtractor()
    
    # Load and parse textgrid file
    textgrid = text_extractor.load_textgrid(textgrid_file)  # Changed from parse_textgrid to load_textgrid
    
    # Get word intervals
    intervals = text_extractor.get_word_intervals(textgrid)
    
    # Align text to audio frames
    text_features = text_extractor.create_text_features(intervals, num_frames)
    
    return audio_frames, text_features.cpu().numpy()


def create_multimodal_windows(audio_frames, text_frames, frames_per_window=100):
    """Create windowed multimodal features"""
    total_frames = audio_frames.shape[0]
    n_windows = total_frames // frames_per_window
    
    audio_windows = [
        audio_frames[i * frames_per_window:(i + 1) * frames_per_window]
        for i in range(n_windows)
    ]
    
    text_windows = [
        text_frames[i * frames_per_window:(i + 1) * frames_per_window]
        for i in range(n_windows)
    ]
    
    return np.stack(audio_windows), np.stack(text_windows)  # shapes: (n_windows, 100, 1024), (n_windows, 100, 768)


print("DONE")


