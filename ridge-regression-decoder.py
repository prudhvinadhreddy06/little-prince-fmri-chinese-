import os
import glob
import shutil
import math
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import subprocess
import re
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import gc
import time
import joblib
from sklearn.metrics import mean_squared_error

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
NUM_EPOCHS = 20  # Not directly used for Ridge Regression but kept for compatibility
PATIENCE = 5     # Not directly used for Ridge Regression but kept for compatibility
BATCH_SIZE = 16  # Increased since Ridge Regression is more memory efficient
LEARNING_RATE = 1e-4  # Used for subsequent neural network fine-tuning if needed

# Memory management settings
MEMORY_CLEANUP_PAUSE = 0.01  # Seconds to pause after memory cleanup

# Ridge Regression Model Class
class RidgeRegressionDecoder:
    def __init__(self, alpha=1.0):
        """
        A Ridge Regression model to decode audio features from fMRI signals
        
        Parameters:
        - alpha: Regularization strength (default=1.0)
        - n_jobs: Number of jobs for parallel processing (-1 uses all processors)
        """
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha, random_state=42, fit_intercept=True, 
                           solver='saga', max_iter=1000, n_jobs=n_jobs))
        ])
        self.feature_dim = None
        self.is_fitted = False
        self.alpha = alpha
        
    def fit(self, X, y):
        """
        Fit the Ridge Regression model
        
        Parameters:
        - X: fMRI data of shape (n_samples, n_voxels)
        - y: Audio features of shape (n_samples, n_features)
        """
        print(f"Fitting Ridge Regression model with alpha={self.alpha}")
        print(f"Input shape: {X.shape}, Output shape: {y.shape}")
        
        self.feature_dim = y.shape[1]
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict audio features from fMRI data
        
        Parameters:
        - X: fMRI data of shape (n_samples, n_voxels)
        
        Returns:
        - Predicted audio features of shape (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.pipeline.predict(X)
    
    def save(self, filepath):
        """Save the model to disk"""
        joblib.dump(self.pipeline, filepath)
        
    def load(self, filepath):
        """Load the model from disk"""
        self.pipeline = joblib.load(filepath)
        self.is_fitted = True
        return self

# Reuse the pearson_correlation_loss function from the encoding model
def pearson_correlation_loss(y_pred, y_true, epsilon=1e-8):
    """
    Calculates 1 - mean(pearson correlation) as a loss function.
    Higher correlation = lower loss.
    Includes safeguards against NaN values.
    """
    # Convert numpy arrays to torch tensors if needed
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
        
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
        return torch.tensor(1.0, device=y_pred.device if isinstance(y_pred, torch.Tensor) else 'cpu')
    
    # Calculate mean correlation only over valid values
    mean_correlation = correlation[valid_mask].mean()
    
    # Convert to loss (1 - correlation)
    loss = 1 - mean_correlation
    
    return loss

# Calculate Pearson correlation coefficient (without loss transformation)
def pearson_correlation(y_pred, y_true, epsilon=1e-8):
    """Calculate Pearson correlation coefficient between predictions and ground truth."""
    loss = pearson_correlation_loss(y_pred, y_true, epsilon)
    if isinstance(loss, torch.Tensor):
        return 1 - loss.item()
    return 1 - loss

# Reuse existing function for loading fMRI data
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

def train_ridge_regression_for_subject(subject_id, matched_files, shared_model=None, alpha=1.0):
    """Train ridge regression model for a single subject with concatenated sections."""
    print(f"\n==== Training ridge regression model for subject {subject_id} ====")
    
    # Use shared model if provided, otherwise create a new one
    if shared_model is None:
        model = RidgeRegressionDecoder(alpha=alpha)
    else:
        model = shared_model
    
    # Section accumulation with memory management
    X_full = None  # Will hold fMRI data as input
    Y_full = None  # Will hold audio features as target
    
    # Process each matched file pair
    for feature_file, fmri_file, section_num, run_num in matched_files:
        print(f"Loading section {section_num}...")
        try:
            # Load and process data
            audio_data = np.load(feature_file, allow_pickle=True).item()
            fmri_data, _ = load_fmri_data(fmri_file)
            
            # Remove last volume to align with audio features
            fmri_data = fmri_data[:-1, :]
            
            # Process audio features
            audio_features = audio_data['frame_features']
            
            # Match lengths to ensure compatibility
            min_length = min(fmri_data.shape[0], audio_features.shape[0])
            fmri_data = fmri_data[:min_length]
            audio_features = audio_features[:min_length]
            
            # Incremental concatenation
            if X_full is None:
                X_full = fmri_data
                Y_full = audio_features
            else:
                X_full = np.concatenate((X_full, fmri_data), axis=0)
                Y_full = np.concatenate((Y_full, audio_features), axis=0)
                
            # Cleanup
            del audio_data, fmri_data, audio_features
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(MEMORY_CLEANUP_PAUSE)
            
        except Exception as e:
            print(f"Error in section {section_num}: {e}")
            continue

    if X_full is None or len(X_full) == 0:
        print("No valid data found!")
        return 0.0, model
        
    # Split data into training and validation sets
    train_size = int(0.8 * X_full.shape[0])
    indices = np.random.permutation(X_full.shape[0])
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    X_train, X_val = X_full[train_indices], X_full[val_indices]
    Y_train, Y_val = Y_full[train_indices], Y_full[val_indices]
    
    print(f"Training set size: {X_train.shape}, Validation set size: {X_val.shape}")
    
    # Fit the model
    try:
        model.fit(X_train, Y_train)
    except Exception as e:
        print(f"Error fitting model: {e}")
        return 0.0, model
    
    # Predict on validation set
    Y_pred = model.predict(X_val)
    
    # Calculate validation correlation
    val_corr = pearson_correlation(Y_pred, Y_val)
    print(f"Validation correlation: {val_corr:.4f}")
    
    # Calculate MSE
    mse = mean_squared_error(Y_val, Y_pred)
    print(f"Validation MSE: {mse:.4f}")
    
    # Save the model
    saved_model_path = f"{MODEL_DIR}/ridge_model_sub-{subject_id}.joblib"
    model.save(saved_model_path)
    print(f"Saved model to {saved_model_path}")
    
    # Save training results
    results = [{
        'subject_id': subject_id,
        'alpha': alpha,
        'train_size': X_train.shape[0],
        'val_size': X_val.shape[0],
        'val_correlation': val_corr,
        'val_mse': mse
    }]
    
    results_df = pd.DataFrame(results)
    results_file = f"{RESULTS_DIR}/ridge_results_sub-{subject_id}.csv"
    results_df.to_csv(results_file, index=False)
    
    # Clean up
    del X_train, X_val, Y_train, Y_val, Y_pred
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return val_corr, model

def evaluate_ridge_model(subject_id, matched_files, model_path):
    """Evaluate a trained ridge regression model on test data."""
    print(f"\n==== Evaluating ridge model for subject {subject_id} ====")
    
    # Load the model
    model = RidgeRegressionDecoder()
    model.load(model_path)
    
    # Store results
    section_results = []
    
    # Process each matched file pair
    for feature_file, fmri_file, section_num, run_num in matched_files:
        print(f"Evaluating section {section_num}...")
        try:
            # Load and process data
            audio_data = np.load(feature_file, allow_pickle=True).item()
            fmri_data, _ = load_fmri_data(fmri_file)
            
            # Remove last volume to align with audio features
            fmri_data = fmri_data[:-1, :]
            
            # Process audio features
            audio_features = audio_data['frame_features']
            
            # Match lengths to ensure compatibility
            min_length = min(fmri_data.shape[0], audio_features.shape[0])
            fmri_data = fmri_data[:min_length]
            audio_features = audio_features[:min_length]
            
            # Predict audio features
            pred_features = model.predict(fmri_data)
            
            # Calculate correlation
            correlation = pearson_correlation(pred_features, audio_features)
            
            # Calculate MSE
            mse = mean_squared_error(audio_features, pred_features)
            
            # Save result
            section_results.append({
                'subject_id': subject_id,
                'section_num': section_num,
                'run_num': run_num,
                'correlation': correlation,
                'mse': mse
            })
            
            # Cleanup
            del audio_data, fmri_data, audio_features, pred_features
            gc.collect()
            
        except Exception as e:
            print(f"Error evaluating section {section_num}: {e}")
            continue
    
    # Save evaluation results
    if section_results:
        results_df = pd.DataFrame(section_results)
        results_file = f"{RESULTS_DIR}/ridge_eval_sub-{subject_id}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Calculate overall average
        avg_corr = results_df['correlation'].mean()
        avg_mse = results_df['mse'].mean()
        print(f"Average correlation for subject {subject_id}: {avg_corr:.4f}")
        print(f"Average MSE for subject {subject_id}: {avg_mse:.4f}")
        
        return avg_corr
    else:
        print(f"No valid evaluation results for subject {subject_id}")
        return float('nan')

def optimize_alpha_for_subject(subject_id, matched_files, alphas=[0.01, 0.1, 1.0, 10.0, 100.0]):
    """Find the optimal regularization parameter (alpha) for ridge regression."""
    print(f"\n==== Optimizing alpha for subject {subject_id} ====")
    
    results = []
    best_corr = -float('inf')
    best_alpha = None
    best_model = None
    
    for alpha in alphas:
        print(f"\nTesting alpha = {alpha}")
        val_corr, model = train_ridge_regression_for_subject(subject_id, matched_files, alpha=alpha)
        
        results.append({
            'subject_id': subject_id,
            'alpha': alpha,
            'val_correlation': val_corr
        })
        
        if val_corr > best_corr:
            best_corr = val_corr
            best_alpha = alpha
            best_model = model
    
    # Save results
    results_df = pd.DataFrame(results)
    results_file = f"{RESULTS_DIR}/ridge_alpha_tuning_sub-{subject_id}.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"Best alpha for subject {subject_id}: {best_alpha} (correlation: {best_corr:.4f})")
    
    return best_alpha, best_corr, best_model

def main():
    """Main function to process all subjects."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting multi-subject ridge regression decoding pipeline at {timestamp}")
    
    # Create a results summary dataframe
    summary_results = []
    
    # Whether to use a shared model across subjects
    use_shared_model = False
    shared_model = None
    
    # Process each subject
    for subj_num in range(1, NUM_SUBJECTS + 1):
        if subj_num == 12 or subj_num == 35:  # Skip problematic subjects
            continue
            
        subject_id = f"CN{subj_num:03d}"
        print(f"\n\n{'='*50}")
        print(f"Processing subject {subject_id} ({subj_num}/{NUM_SUBJECTS})")
        print(f"{'='*50}")
        
        # Download subject data
        if download_subject_data(subject_id):
            # Get matching files
            matched_files = get_matching_files(subject_id)
            
            if matched_files:
                # Option 1: Simple training with default alpha
                # val_corr, model = train_ridge_regression_for_subject(subject_id, matched_files, shared_model if use_shared_model else None)
                
                # Option 2: Alpha tuning for better results
                best_alpha, val_corr, model = optimize_alpha_for_subject(subject_id, matched_files)
                
                # Update shared model if using one
                if use_shared_model:
                    shared_model = model
                
                # Add results to summary
                summary_results.append({
                    'subject_id': subject_id,
                    'num_paired_files': len(matched_files),
                    'best_val_correlation': val_corr,
                    'best_alpha': best_alpha if 'best_alpha' in locals() else 1.0  # Default if not using alpha tuning
                })
                
                # Evaluate on all sections
                model_path = f"{MODEL_DIR}/ridge_model_sub-{subject_id}.joblib"
                avg_corr = evaluate_ridge_model(subject_id, matched_files, model_path)
                
                # Update summary with evaluation results
                summary_results[-1]['test_correlation'] = avg_corr
            
            # Clean up subject data to save space
            clean_subject_data(subject_id)
            
    # Save summary results
    summary_df = pd.DataFrame(summary_results)
    summary_file = f"{RESULTS_DIR}/ridge_decoding_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\nRidge regression decoding complete! Summary saved to {summary_file}")
    print(f"Average validation correlation across subjects: {summary_df['best_val_correlation'].mean():.4f}")
    if 'test_correlation' in summary_df.columns:
        print(f"Average test correlation across subjects: {summary_df['test_correlation'].mean():.4f}")

if __name__ == "__main__":
    main()
