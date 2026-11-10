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



def download_subject_data(subject_id):
    """Download a subject's data using openneuro-py"""
    subj_dir = f"{SUBJECTS_DIR}/sub-{subject_id}"
    
    # Skip download if data already exists
    if os.path.exists(f"{subj_dir}/derivatives/sub-{subject_id}/func"):
        print(f"Data for subject {subject_id} already exists.")
        return True
    
    print(f"Downloading data for subject {subject_id}...")
    
    try:
        # Create subject directory if it doesn't exist
        os.makedirs(subj_dir, exist_ok=True)
        
        # Download command using openneuro-py
        cmd = [
            "openneuro-py",
            "download",
            "--dataset", "ds003643",
            "--include", f"derivatives/sub-{subject_id}/*",
            "--target-dir", subj_dir,
            "--verify-hash"
        ]
        
        # Run the download command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Download failed for subject {subject_id}:")
            print(result.stderr)
            return False
            
        # Verify the downloaded files
        required_files = glob.glob(f"{subj_dir}/derivatives/sub-{subject_id}/func/*.nii.gz")
        if not required_files:
            print(f"No fMRI files found for subject {subject_id} after download")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error downloading data for subject {subject_id}: {str(e)}")
        return False

print("DONE")



def load_fmri_data(func_file, brain_mask=None):
    """Load fMRI data from a NIfTI file."""
    try:
        fmri_img = nib.load(func_file)
        data = fmri_img.get_fdata()  # shape: (x, y, z, t)
        tr = fmri_img.header.get_zooms()[3]
        n_volumes = data.shape[3]

        if brain_mask is not None:
            mask = nib.load(brain_mask).get_fdata().astype(bool)
            data = data[mask]  # shape: (n_voxels, t)
            data = data.T  # shape: (t, n_voxels)
        else:
            # To save memory, process the data in chunks
            original_shape = data.shape
            # Reshape more efficiently without loading entire array into new memory
            data = np.reshape(data, (-1, n_volumes)).T  # shape: (t, voxels)
        
        return data, tr
    except Exception as e:
        print(f"Error loading fMRI data: {e}")
        raise

print("DONE")



def pearson_correlation_loss(y_pred, y_true, epsilon=1e-8):
    """
    Calculates 1 - mean(pearson correlation) as a loss function.
    Higher correlation = lower loss.
    Includes safeguards against NaN values.
    """
    # Check if inputs contain NaN
    if torch.isnan(y_pred).any() or torch.isnan(y_true).any():
        print("Warning: NaN detected in inputs to correlation loss")
        # Replace NaNs with zeros
        y_pred = torch.nan_to_num(y_pred, nan=0.0)
        y_true = torch.nan_to_num(y_true, nan=0.0)
    
    # Center the data (subtract mean)
    y_pred_centered = y_pred - y_pred.mean(dim=0, keepdim=True)
    y_true_centered = y_true - y_true.mean(dim=0, keepdim=True)
    
    # Calculate numerator and denominator
    numerator = torch.sum(y_pred_centered * y_true_centered, dim=0)
    
    # Calculate standard deviations with epsilon to avoid division by zero
    y_pred_std = torch.sqrt(torch.sum(y_pred_centered**2, dim=0) + epsilon)
    y_true_std = torch.sqrt(torch.sum(y_true_centered**2, dim=0) + epsilon)
    
    # Calculate correlation
    correlation = numerator / (y_pred_std * y_true_std)
    
    # Clamp correlation to valid range [-1, 1]
    correlation = torch.clamp(correlation, min=-1.0, max=1.0)
    
    # Mask out any remaining NaN values (should be rare with proper handling above)
    valid_mask = ~torch.isnan(correlation)
    if valid_mask.sum() == 0:
        # If all correlations are NaN, return a default value
        return torch.tensor(1.0, device=y_pred.device)
    
    # Calculate mean correlation only over valid values
    mean_correlation = correlation[valid_mask].mean()
    
    # Convert to loss (1 - correlation)
    loss = 1 - mean_correlation
    
    return loss


def clean_subject_data(subject_id):
    """Remove subject data to free up space."""
    subj_dir = f"{SUBJECTS_DIR}/sub-{subject_id}"
    
    if os.path.exists(subj_dir):
        print(f"Cleaning up data for subject {subject_id}...")
        shutil.rmtree(subj_dir)


def train_multimodal_model(subject_id, matched_files, model=None, optimizer=None):
    """Train multimodal model for a single subject"""
    print(f"\n==== Training multimodal model for subject {subject_id} ====")
    
    # Initialize model if not provided
    if model is None:
        # Get output dimension from first fMRI file
        first_file = matched_files[0]
        _, _, fmri_file, _, _ = first_file
        test_fmri, _ = load_fmri_data(fmri_file)
        output_dim = test_fmri.shape[1]
        
        model = MultimodalEncoder(output_dim=output_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize training components
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    best_val_loss = float('inf')
    patience_counter = 0
    saved_model_path = f"{MODEL_DIR}/best_model_shared_multimodal.pt"
    results = []
    
    # Initialize text feature extractor (shared across subjects)
    text_extractor = TextFeatureExtractor()
    
    # Accumulate data across sections
    X_audio_full = None
    X_text_full = None
    Y_full = None
    
    for feature_file, textgrid_file, fmri_file, section_num, run_num in matched_files:
        print(f"Loading section {section_num}...")
        try:
            # Load fMRI data
            Y_fmri, _ = load_fmri_data(fmri_file)
            Y_fmri = Y_fmri[:-1, :]  # Remove last volume
            
            # Load and align multimodal data
            audio_frames, text_frames = load_multimodal_data(feature_file, textgrid_file)
            
            # Create windows
            X_audio, X_text = create_multimodal_windows(audio_frames, text_frames)
            
            # Verify alignment
            if X_audio.shape[0] != Y_fmri.shape[0]:
                print(f"Shape mismatch in section {section_num} (audio: {X_audio.shape[0]}, fMRI: {Y_fmri.shape[0]}), skipping")
                continue
                
            # Convert to tensors
            X_audio_tensor = torch.tensor(X_audio, dtype=torch.float32)
            X_text_tensor = torch.tensor(X_text, dtype=torch.float32)
            Y_tensor = torch.tensor(Y_fmri, dtype=torch.float32)
            
            # Incremental concatenation
            if X_audio_full is None:
                X_audio_full = X_audio_tensor
                X_text_full = X_text_tensor
                Y_full = Y_tensor
            else:
                X_audio_full = torch.cat((X_audio_full, X_audio_tensor), dim=0)
                X_text_full = torch.cat((X_text_full, X_text_tensor), dim=0)
                Y_full = torch.cat((Y_full, Y_tensor), dim=0)
                
            # Cleanup
            del audio_frames, text_frames, X_audio, X_text, Y_fmri, X_audio_tensor, X_text_tensor, Y_tensor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(MEMORY_CLEANUP_PAUSE)
            
        except Exception as e:
            print(f"Error in section {section_num}: {str(e)}")
            continue

    if X_audio_full is None:
        print("No valid data found for training!")
        return 0.0, model, optimizer
        
    # Create dataset and dataloaders
    dataset = TensorDataset(X_audio_full, X_text_full, Y_full)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    num_workers = 4 if torch.cuda.is_available() else 2
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        # Training phase
        for x_audio, x_text, yb in tqdm(train_loader, desc="Training"):
            x_audio = x_audio.to(device, non_blocking=True)
            x_text = x_text.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            pred = model(x_audio, x_text)
            loss = pearson_correlation_loss(pred, yb)
            
            if torch.isnan(loss):
                print("NaN loss detected, skipping batch")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            del x_audio, x_text, yb, pred, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x_audio, x_text, yb in tqdm(val_loader, desc="Validation"):
                x_audio = x_audio.to(device)
                x_text = x_text.to(device)
                yb = yb.to(device)
                
                pred = model(x_audio, x_text)
                loss = pearson_correlation_loss(pred, yb)
                
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    val_batches += 1
                
                del x_audio, x_text, yb, pred, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Calculate metrics
        avg_train_loss = train_loss / train_batches if train_batches > 0 else float('nan')
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('nan')
        train_corr = 1 - avg_train_loss if not math.isnan(avg_train_loss) else float('nan')
        val_corr = 1 - avg_val_loss if not math.isnan(avg_val_loss) else float('nan')
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss = {avg_train_loss:.4f}, Train Corr = {train_corr:.4f}")
        print(f"Val Loss = {avg_val_loss:.4f}, Val Corr = {val_corr:.4f}")
        print(f"LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # Update scheduler
        if not math.isnan(avg_val_loss):
            scheduler.step(avg_val_loss)
        
        # Save results
        results.append({
            'subject_id': subject_id,
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_corr': train_corr,
            'val_loss': avg_val_loss,
            'val_corr': val_corr,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Early stopping check
        if not math.isnan(avg_val_loss):
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), saved_model_path)
                print(f"Saved new best model with val loss: {best_val_loss:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print("Learning rate too small, stopping training")
            break
    
    # Save training results
    results_df = pd.DataFrame(results)
    results_file = f"{RESULTS_DIR}/training_results_multimodal_sub-{subject_id}.csv"
    results_df.to_csv(results_file, index=False)
    
    best_val_corr = 1 - best_val_loss
    return best_val_corr, model, optimizer



def get_matching_files_with_text(subject_id):
    """Get matching feature, textgrid and fMRI files for a subject"""
    # Get all feature files
    feature_files = sorted(glob.glob(f"{FEATURES_DIR}/task-lppCN_section_*_features.npy"))
    
    # Get all textgrid files (assuming same naming convention with .TextGrid extension)
    text_grid_files_location = "/kaggle/working/ds003643-download/annotations/annotation/CN"
    textgrid_files = sorted(glob.glob(f"{text_grid_files_location}/lppCN_section*.TextGrid"))
    
    # Get all fMRI files for this subject
    fmri_pattern = f"{SUBJECTS_DIR}/sub-{subject_id}/derivatives/sub-{subject_id}/func/sub-{subject_id}_task-lppCN_run-*_space-MNIColin27_desc-preproc_bold.nii.gz"
    fmri_files = sorted(glob.glob(fmri_pattern))
    
    print(f"\nFile counts for subject {subject_id}:")
    print(f"Feature files: {len(feature_files)}")
    print(f"TextGrid files: {len(textgrid_files)}")
    print(f"fMRI files: {len(fmri_files)}")
    
    # Verify we have matching sets
    if not feature_files or not textgrid_files or not fmri_files:
        print(f"Missing files for subject {subject_id} - skipping")
        return []
    
    # Extract section numbers with more robust parsing
    def extract_number(filename, pattern):
        match = re.search(pattern, filename)
        return int(match.group(1)) if match else -1
    
    # Create lists of (section_num, file_path) with valid files only
    feature_sections = []
    for f in feature_files:
        section_num = extract_number(f, r'section_(\d+)_features')
        if section_num != -1:
            feature_sections.append((section_num, f))
    
    textgrid_sections = []
    for f in textgrid_files:
        section_num = extract_number(f, r'section(\d+)')
        if section_num != -1:
            textgrid_sections.append((section_num, f))
    
    fmri_runs = []
    for f in fmri_files:
        run_num = extract_number(f, r'run-(\d+)_')
        if run_num != -1:
            fmri_runs.append((run_num, f))
    
    # Sort all lists by section/run number
    feature_sections.sort()
    textgrid_sections.sort()
    fmri_runs.sort()
    
    min_length = min(len(feature_sections), len(fmri_runs) , len(textgrid_sections))


    matched_files = []
    for i in range(min_length):
        section_num, feature_file = feature_sections[i]
        run_num, fmri_file = fmri_runs[i]
        textgrid_num , text_grid_file = textgrid_sections[i]
        matched_files.append((feature_file,text_grid_file, fmri_file, section_num, run_num))
    
    print(f"Found {len(matched_files)} matching file pairs for subject {subject_id}")
    
    return matched_files


def main():
    """Main function to process all subjects with multimodal data"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting multimodal training pipeline at {timestamp}")
    
    # Create a results summary dataframe
    summary_results = []
    
    # Initialize shared model and optimizer
    model = None
    optimizer = None
    
    # Process each subject
    for subj_num in range(1, NUM_SUBJECTS + 1):
        if subj_num in [12, 35]:  # Skip problematic subjects
            continue
            
        subject_id = f"CN{subj_num:03d}"
        print(f"\n\n{'='*50}")
        print(f"Processing subject {subject_id} ({subj_num}/{NUM_SUBJECTS})")
        print(f"{'='*50}")

        
        
        # Download subject data
        if download_subject_data(subject_id):
            # Get matching files - now including textgrid files
            matched_files = get_matching_files_with_text(subject_id)
            
            if matched_files:
                # Train model for this subject
                best_corr, model, optimizer = train_multimodal_model(
                    subject_id, matched_files, model, optimizer)
                
                # Add results to summary
                summary_results.append({
                    'subject_id': subject_id,
                    'num_paired_files': len(matched_files),
                    'best_val_correlation': best_corr
                })
            
            # Clean up subject data to save space
            clean_subject_data(subject_id)
    
    # Save summary results
    summary_df = pd.DataFrame(summary_results)
    summary_file = f"{RESULTS_DIR}/training_summary_multimodal_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\nTraining complete! Summary saved to {summary_file}")
    if len(summary_results) > 0:
        avg_corr = summary_df['best_val_correlation'].mean()
        print(f"Average validation correlation across subjects: {avg_corr:.4f}")
if __name__ == "__main__":
    main()