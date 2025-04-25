import torch
import torch.nn as nn
import numpy as np
import hashlib
import os
import time
from torch.utils.data import DataLoader, Subset

def sanitize_data(dataset, threshold=2.0, per_feature=True):
    """
    Detects and removes potentially poisoned data from the dataset.
    Uses statistical outlier detection on a per-feature basis if enabled.
    
    Args:
        dataset: The dataset to sanitize
        threshold: Outlier threshold in standard deviations
        per_feature: Whether to check outliers per feature (True) or whole samples (False)
    
    Returns:
        Sanitized dataset
    """
    x_data = dataset.data
    if isinstance(x_data, torch.Tensor):
        x_data = x_data.numpy()
    
    # Convert labels to numpy if needed
    y_data = dataset.targets
    if isinstance(y_data, torch.Tensor):
        y_data = y_data.numpy()
    elif isinstance(y_data, list):
        y_data = np.array(y_data)
    
    # Flatten images if needed
    orig_shape = x_data.shape
    if len(orig_shape) > 2:
        if per_feature:
            # Keep samples as first dimension, flatten the rest
            x_flat = x_data.reshape(orig_shape[0], -1)
        else:
            # Keep original shape for whole-sample comparison
            x_flat = x_data
    else:
        x_flat = x_data
    
    # Calculate statistics
    mean = np.mean(x_flat, axis=0)
    std = np.std(x_flat, axis=0)
    std = np.where(std == 0, 1e-6, std)  # Avoid division by zero
    
    # Find clean samples
    clean_indices = []
    
    if per_feature:
        # Per-feature anomaly detection
        z_scores = np.abs((x_flat - mean) / std)
        max_z_scores = np.max(z_scores, axis=1)
        clean_indices = np.where(max_z_scores < threshold)[0]
    else:
        # Whole sample anomaly detection
        for i, sample in enumerate(x_flat):
            if np.linalg.norm((sample - mean) / std) < threshold:
                clean_indices.append(i)
    
    print(f"Sanitization removed {len(x_data) - len(clean_indices)} samples ({(len(x_data) - len(clean_indices)) / len(x_data) * 100:.2f}%)")
    
    # Create sanitized dataset
    if hasattr(dataset, 'data') and hasattr(dataset, 'targets'):
        dataset.data = x_data[clean_indices]
        dataset.targets = y_data[clean_indices]
    else:
        # For datasets without direct access to data and targets
        # Create a subset using indices
        return Subset(dataset, clean_indices)
    
    return dataset

def compute_checkpoint_hash(checkpoint_path):
    """
    Computes MD5 hash of checkpoint file.
    """
    hasher = hashlib.md5()
    with open(checkpoint_path, "rb") as f:
        buf = f.read(65536)  # Read in 64k chunks
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def store_checkpoint_hash(checkpoint_path, hash_dir="./checkpoint_hashes"):
    """
    Stores the hash of a checkpoint in a separate file.
    """
    os.makedirs(hash_dir, exist_ok=True)
    
    checkpoint_hash = compute_checkpoint_hash(checkpoint_path)
    hash_file = os.path.join(hash_dir, os.path.basename(checkpoint_path) + ".hash")
    
    with open(hash_file, "w") as f:
        f.write(checkpoint_hash)
    
    return checkpoint_hash

def validate_checkpoint(checkpoint_path, hash_dir="./checkpoint_hashes"):
    """
    Ensures the checkpoint file has not been tampered with.
    Uses hash comparison from a separate hash store.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        hash_dir: Directory storing hash files
        
    Returns:
        Boolean indicating if checkpoint is valid
    """
    if not os.path.exists(checkpoint_path):
        print("Checkpoint does not exist!")
        return False
    
    hash_file = os.path.join(hash_dir, os.path.basename(checkpoint_path) + ".hash")
    
    if not os.path.exists(hash_file):
        print("No hash file found. Creating new hash.")
        store_checkpoint_hash(checkpoint_path, hash_dir)
        return True
    
    with open(hash_file, "r") as f:
        expected_hash = f.read().strip()
    
    current_hash = compute_checkpoint_hash(checkpoint_path)
    
    if current_hash != expected_hash:
        print("WARNING: Checkpoint hash mismatch detected!")
        return False
    
    return True



def create_model_backup(checkpoint_path, backup_dir="./model_backups"):
    """
    Creates a backup of the model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        backup_dir: Directory to store backups
    
    Returns:
        Path to the backup file
    """
    os.makedirs(backup_dir, exist_ok=True)
    
    # Create a timestamped backup
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_file = os.path.join(
        backup_dir, 
        f"{os.path.basename(checkpoint_path)}.{timestamp}.bak"
    )
    
    # Copy the checkpoint
    import shutil
    shutil.copy2(checkpoint_path, backup_file)
    
    # Store the hash
    store_checkpoint_hash(backup_file)
    
    print(f"Created backup at {backup_file}")
    return backup_file

def get_latest_backup(checkpoint_name, backup_dir="./model_backups"):
    """
    Gets the most recent valid backup for a checkpoint.
    
    Args:
        checkpoint_name: Base name of the checkpoint
        backup_dir: Directory with backups
        
    Returns:
        Path to the latest valid backup, or None if no valid backup exists
    """
    if not os.path.exists(backup_dir):
        return None
    
    # Find all backups for this checkpoint
    backups = [f for f in os.listdir(backup_dir) 
              if f.startswith(checkpoint_name) and f.endswith(".bak")]
    
    if not backups:
        return None
    
    # Sort by timestamp (newest first)
    backups.sort(reverse=True)
    
    # Find first valid backup
    for backup in backups:
        backup_path = os.path.join(backup_dir, backup)
        if validate_checkpoint(backup_path):
            return backup_path
    
    return None

def rollback_model(checkpoint_path, backup_dir="./model_backups"):
    """
    Restores model from a clean backup if checkpoint is corrupted.
    
    Args:
        checkpoint_path: Path to potentially corrupted checkpoint
        backup_dir: Directory with backups
        
    Returns:
        Boolean indicating if rollback was successful
    """
    if validate_checkpoint(checkpoint_path):
        print("Checkpoint is valid, no rollback needed.")
        return True
    
    print("Corrupted checkpoint detected! Rolling back...")
    
    checkpoint_name = os.path.basename(checkpoint_path)
    latest_backup = get_latest_backup(checkpoint_name, backup_dir)
    
    if latest_backup:
        import shutil
        shutil.copy2(latest_backup, checkpoint_path)
        print(f"Model restored from backup: {latest_backup}")
        return True
    else:
        print("No valid backup found. Cannot rollback.")
        return False

def noise_detection(model, original_model, threshold=0.1):
    """
    Detects if a model has been potentially poisoned with BrainWash noise.
    Compares parameter distributions with an original clean model.
    
    Args:
        model: The potentially poisoned model
        original_model: A known clean model (or baseline statistics)
        threshold: Threshold for detecting anomalous parameters
        
    Returns:
        Boolean indicating if poisoning is detected
    """
    anomaly_detected = False
    
    for (name1, param1), (name2, param2) in zip(
        model.named_parameters(), original_model.named_parameters()
    ):
        # Check parameter names match
        if name1 != name2:
            continue
            
        # Compute absolute difference
        diff = torch.abs(param1 - param2)
        
        # Check for unusually large differences
        if diff.max() > threshold:
            print(f"Anomaly detected in {name1}: max diff = {diff.max().item()}")
            anomaly_detected = True
    
    return anomaly_detected

def mitigate_brainwash_attack(model, dataloader, device='cuda'):
    """
    Attempt to mitigate BrainWash attack effects by applying
    noise filtering and parameter regularization.
    
    Args:
        model: Model potentially affected by BrainWash
        dataloader: Data loader with clean samples
        device: Computation device
        
    Returns:
        Mitigated model
    """
    model.to(device)
    model.eval()
    
    # 1. Parameter smoothing via moving average
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Skip certain layers if needed
            if 'batch_norm' in name:
                continue
                
            # Apply subtle smoothing to parameters
            # This helps reduce potential noise patterns while preserving function
            smooth_param = torch.nn.functional.avg_pool1d(
                param.view(1, -1, 1), kernel_size=3, stride=1, padding=1
            ).view(param.size())
            
            # Update with smoothed version
            param.copy_(smooth_param)
    
    # 2. Loss landscape analysis - detect and fix irregular sharpness
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass with noise
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip extremely large gradients - these might be from poisoning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Apply update
        optimizer.step()
    
    return model