!pip install nilearn openneuro-py

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




# Model definition
class FrameSequenceEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=1000):
        super().__init__()
        
        # Temporal convolutional processing
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Recurrent processing
        self.gru = nn.GRU(hidden_dim, hidden_dim//2, 
                          batch_first=False, 
                          bidirectional=True,
                          num_layers=2,
                          dropout=0.2)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # Output mapping
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # x shape: (batch, time, features)

        x = x.permute(0, 2, 1)  # -> (batch, features, time)

        # Convolutional processing
        x = self.conv_layers(x)  # -> (batch, hidden, time)
      
        # Prepare for RNN
        x = x.permute(2, 0, 1)  # -> (time, batch, hidden)
  
        # RNN processing
        rnn_out, _ = self.gru(x)  # -> (time, batch, hidden)
        rnn_out = rnn_out.permute(1, 0, 2)  # -> (batch, time, hidden)
      
        # Attention
        attn_weights = self.attention(rnn_out).squeeze(-1)  # -> (batch, time)
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)  # -> (batch, 1, time)
       
        # Apply attention
        context = torch.bmm(attn_weights, rnn_out).squeeze(1)  # -> (batch, hidden)
      
        # Project to output space
        output = self.output(context)  # -> (batch, output_dim)
      
        return output

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
def create_frame_windows(frame_features, frames_per_window=100):
    """Create windowed feature frames for processing."""
    total_frames = frame_features.shape[0]
    n_windows = total_frames // frames_per_window
    windows = [
        frame_features[i * frames_per_window:(i + 1) * frames_per_window]
        for i in range(n_windows)
    ]
    return np.stack(windows)  # shape (n_windows, 100, 1024)

def download_subject_data(subject_id):
    """Download a subject's data using openneuro-py."""
    subj_dir = f"{SUBJECTS_DIR}/sub-{subject_id}"
    
    # Skip download if data already exists
    if os.path.exists(f"{subj_dir}/derivatives/sub-{subject_id}/func"):
        print(f"Data for subject {subject_id} already exists.")
        return True
    
    print(f"Downloading data for subject {subject_id}...")
    cmd = f"openneuro-py download --dataset ds003643 --include \"derivatives/sub-{subject_id}/*\" --target-dir \"{SUBJECTS_DIR}/sub-{subject_id}\" --verify-hash"
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading data for subject {subject_id}: {e}")
        return False

def clean_subject_data(subject_id):
    """Remove subject data to free up space."""
    subj_dir = f"{SUBJECTS_DIR}/sub-{subject_id}"
    
    if os.path.exists(subj_dir):
        print(f"Cleaning up data for subject {subject_id}...")
        shutil.rmtree(subj_dir)

def get_matching_files(subject_id):
    """Get matching feature and fMRI files for a subject."""
    # Get all feature files
    feature_files = sorted(glob.glob(f"{FEATURES_DIR}/task-lppCN_section_*_features.npy"))
    
    # Get all fMRI files for this subject
    fmri_pattern = f"{SUBJECTS_DIR}/sub-{subject_id}/derivatives/sub-{subject_id}/func/sub-{subject_id}_task-lppCN_run-*_space-MNIColin27_desc-preproc_bold.nii.gz"
    fmri_files = sorted(glob.glob(fmri_pattern))
    
    if len(feature_files) == 0:
        print("No feature files found!")
        return []
    
    if len(fmri_files) == 0:
        print(f"No fMRI files found for subject {subject_id}!")
        return []
    
    # Match feature files with fMRI files
    # We need to extract run numbers and section numbers to match them properly
    feature_sections = []
    for f in feature_files:
        match = re.search(r'section_(\d+)_features', f)
        if match:
            section = int(match.group(1))
            feature_sections.append((section, f))
    
    fmri_runs = []
    for f in fmri_files:
        match = re.search(r'run-(\d+)_', f)
        if match:
            run = int(match.group(1))
            fmri_runs.append((run, f))
    
    # Print information about found files
    print(f"Found {len(feature_sections)} feature files and {len(fmri_runs)} fMRI files")
    print("Feature sections:", [s[0] for s in feature_sections])
    print("fMRI runs:", [r[0] for r in fmri_runs])
    
    # Sort both lists
    feature_sections.sort()
    fmri_runs.sort()
    
    # Match files based on index (assuming they align correctly when sorted)
    # Take the minimum length to avoid index errors
    min_length = min(len(feature_sections), len(fmri_runs))
    
    matched_files = []
    for i in range(min_length):
        section_num, feature_file = feature_sections[i]
        run_num, fmri_file = fmri_runs[i]
        matched_files.append((feature_file, fmri_file, section_num, run_num))
    
    print(f"Found {len(matched_files)} matching file pairs for subject {subject_id}")
    return matched_files

def train_model_for_subject(subject_id, matched_files, model=None, optimizer=None):
    """Train model for a single subject with concatenated sections using a shared model."""
    print(f"\n==== Training model for subject {subject_id} ====")
    
    # Initialize model if not provided (shared model case)
    if model is None:
        first_file = matched_files[0]
        feature_file, fmri_file, section_num, run_num = first_file
        test_fmri, tr = load_fmri_data(fmri_file)
        output_dim = test_fmri.shape[1]
        
        model = FrameSequenceEncoder(input_dim=1024, output_dim=output_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize scheduler and training components
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    best_val_loss = float('inf')
    patience_counter = 0
    saved_model_path = f"{MODEL_DIR}/best_model_shared.pt"  # Single model for all subjects
    results = []

    # Section accumulation with memory management
    X_full = None
    Y_full = None
    
    for feature_file, fmri_file, section_num, run_num in matched_files:
        print(f"Loading section {section_num}...")
        try:
            # Load and process data
            data = np.load(feature_file, allow_pickle=True).item()
            Y_fmri, _ = load_fmri_data(fmri_file)
            Y_fmri = Y_fmri[:-1, :]  # Remove last volume
            X_audio = create_frame_windows(data['frame_features'])
            
            if X_audio.shape[0] != Y_fmri.shape[0]:
                print(f"Shape mismatch in section {section_num}, skipping")
                continue
                
            # Convert to tensors
            X_tensor = torch.tensor(X_audio, dtype=torch.float32)
            Y_tensor = torch.tensor(Y_fmri, dtype=torch.float32)
            
            # Incremental concatenation
            if X_full is None:
                X_full = X_tensor
                Y_full = Y_tensor
            else:
                X_full = torch.cat((X_full, X_tensor), dim=0)
                Y_full = torch.cat((Y_full, Y_tensor), dim=0)
                
            # Cleanup
            del data, Y_fmri, X_audio, X_tensor, Y_tensor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(MEMORY_CLEANUP_PAUSE)
            
        except Exception as e:
            print(f"Error in section {section_num}: {e}")
            continue

    if X_full is None:
        print("No valid data found!")
        return 0.0, model, optimizer
        
    # Create dataset and dataloaders
    dataset = TensorDataset(X_full, Y_full)
  
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset)-train_size])
    
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
        for xb, yb in tqdm(train_loader, desc="Training"):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
      
            optimizer.zero_grad()
            pred = model(xb)
            loss = pearson_correlation_loss(pred, yb)
            
            if torch.isnan(loss):
                print("NaN loss detected, skipping batch")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            del xb, yb, pred, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc="Validation"):
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = pearson_correlation_loss(pred, yb)
                
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    val_batches += 1
                
                del xb, yb, pred, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Calculate metrics
        avg_train_loss = train_loss / train_batches if train_batches > 0 else float('nan')
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('nan')
        train_corr = 1 - avg_train_loss if not math.isnan(avg_train_loss) else float('nan')
        val_corr = 1 - avg_val_loss if not math.isnan(avg_val_loss) else float('nan')
        
        print(f"Epoch {epoch+1} Summary: Train Loss = {avg_train_loss:.4f}, Train Corr = {train_corr:.4f}, "
              f"Val Loss = {avg_val_loss:.4f}, Val Corr = {val_corr:.4f}, "
              f"LR = {optimizer.param_groups[0]['lr']:.6f}")
        
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
        if not math.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
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
    results_file = f"{RESULTS_DIR}/training_results_sub-{subject_id}.csv"
    results_df.to_csv(results_file, index=False)
    
    best_val_corr = 1 - best_val_loss
    return best_val_corr, model, optimizer



def main():
    """Main function to process all subjects."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting multi-subject training pipeline at {timestamp}")
    
    # Create a results summary dataframe
    summary_results = []

    model = None
    optimizer = None
    # Process each subject
    for subj_num in range(1, NUM_SUBJECTS + 1):
        subject_id = f"CN{subj_num:03d}"
        print(f"\n\n{'='*50}")
        print(f"Processing subject {subject_id} ({subj_num}/{NUM_SUBJECTS})")
        print(f"{'='*50}")
        
        # Download subject data
        if download_subject_data(subject_id):
            # Get matching files
            matched_files = get_matching_files(subject_id)
            
            if matched_files:
                # Train model for this subject

                
                best_corr, model, optimizer = train_model_for_subject(subject_id, matched_files, model, optimizer)
                
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
    summary_file = f"{RESULTS_DIR}/training_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\nTraining complete! Summary saved to {summary_file}")
    print(f"Average validation correlation across subjects: {summary_df['best_val_correlation'].mean():.4f}")

if __name__ == "__main__":
    main()