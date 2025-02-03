# data_prepare.py
import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, vrange

# ! X shape: (B, T, N, C)

def generate_fourier_features(time_of_day, num_harmonics=4):
    """Generate Fourier features for time of day values.
    
    Args:
        time_of_day: normalized time of day values (0-1)
        num_harmonics: number of sin/cos pairs to generate
    
    Returns:
        Fourier features of shape (..., num_harmonics*2)
    """
    # Get original shape to reshape output later
    original_shape = time_of_day.shape
    time_of_day = time_of_day.flatten()
    
    # Generate features
    coeffs = 2 * np.pi * np.arange(1, num_harmonics + 1)
    sin_features = np.vstack([np.sin(c * time_of_day) for c in coeffs]).T 
    cos_features = np.vstack([np.cos(c * time_of_day) for c in coeffs]).T
    
    # Combine and reshape
    fourier_features = np.concatenate([sin_features, cos_features], axis=-1)
    new_shape = original_shape[:-1] + (num_harmonics * 2,)
    return fourier_features.reshape(new_shape)

def get_dataloaders_from_index_data(
    data_dir, tod=False, dow=False, dom=False, num_harmonics=4, batch_size=64, log=None
):
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)

    features = [0]
    if tod:
        features.append(1)
        # Generate Fourier features for time of day
        tod_data = data[..., 1:2]  # Get time of day features
        fourier_features = generate_fourier_features(tod_data, num_harmonics)
        # Append Fourier features to data
        original_data = data.copy()  # Make a copy of original data
        data = np.concatenate([original_data, fourier_features], axis=-1)
    
    if dow:
        features.append(2)

    # Select required features including the original features and Fourier features
    all_features = features.copy()
    if tod:
        all_features.extend(range(original_data.shape[-1], data.shape[-1]))
    data = data[..., all_features]

    index = np.load(os.path.join(data_dir, "index.npz"))

    train_index = index["train"]  # (num_samples, 3)
    val_index = index["val"]
    test_index = index["test"]

    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    x_train = data[x_train_index]
    y_train = data[y_train_index][..., :1]
    x_val = data[x_val_index]
    y_val = data[y_val_index][..., :1]
    x_test = data[x_test_index]
    y_test = data[y_test_index][..., :1]

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())
    
    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0]) 
    x_test[..., 0] = scaler.transform(x_test[..., 0])
    

    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler

def get_k_fold_dataloaders(data_dir, k_folds=5, tod=False, dow=False, dom=False, num_harmonics=4, batch_size=64, log=None):
    """Get K-fold cross validation dataloaders"""
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)
    index = np.load(os.path.join(data_dir, "index.npz"))
    
    # Combine train and val indices
    all_train_index = np.concatenate([index["train"], index["val"]], axis=0)
    
    # Create k folds
    fold_size = len(all_train_index) // k_folds
    fold_dataloaders = []
    
    for fold in range(k_folds):
        # Split indices for this fold
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size
        val_indices = all_train_index[val_start:val_end]
        train_indices = np.concatenate([
            all_train_index[:val_start],
            all_train_index[val_end:]
        ])
        
        # Get data for this fold
        x_train_index = vrange(train_indices[:, 0], train_indices[:, 1])
        y_train_index = vrange(train_indices[:, 1], train_indices[:, 2]) 
        x_val_index = vrange(val_indices[:, 0], val_indices[:, 1])
        y_val_index = vrange(val_indices[:, 1], val_indices[:, 2])
        
        # Process features same as original function
        features = [0]
        if tod:
            features.append(1)
            tod_data = data[..., 1:2]
            fourier_features = generate_fourier_features(tod_data, num_harmonics)
            data_with_fourier = np.concatenate([data, fourier_features], axis=-1)
        if dow:
            features.append(2)
            
        # Get data slices
        x_train = data_with_fourier[x_train_index] if tod else data[x_train_index]
        y_train = data[y_train_index][..., :1]
        x_val = data_with_fourier[x_val_index] if tod else data[x_val_index] 
        y_val = data[y_val_index][..., :1]
        
        # Create scaler and transform data
        scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())
        x_train[..., 0] = scaler.transform(x_train[..., 0])
        x_val[..., 0] = scaler.transform(x_val[..., 0])
        
        # Create dataloaders
        trainset = torch.utils.data.TensorDataset(
            torch.FloatTensor(x_train), torch.FloatTensor(y_train)
        )
        valset = torch.utils.data.TensorDataset(
            torch.FloatTensor(x_val), torch.FloatTensor(y_val)
        )
        
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=False
        )
        
        fold_dataloaders.append((train_loader, val_loader, scaler))
        
    print_log(f"Created {k_folds} cross validation folds", log=log)
    return fold_dataloaders