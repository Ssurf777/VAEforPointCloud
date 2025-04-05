import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from file_io import read_off
from sampling import PointSampler
import pandas as pd

def prepare_data_from_csv(file_names, num_points=5000, device='cuda'):
    """
    Loads and preprocesses point cloud data from CSV files.
    Each CSV file must contain columns: 'X(m)', 'Y(m)', 'Z(m)'.
    
    Args:
        file_names (list of str): List of CSV file paths.
        num_points (int): Number of points to sample from each file (if more are available).
        device (str): 'cuda' or 'cpu'.
        
    Returns:
        DataLoader: DataLoader containing normalized point cloud tensors.
    """
    input_data_list = []

    for file_name in file_names:
        df = pd.read_csv(file_name)
        coords = df[['X(m)', 'Y(m)', 'Z(m)']].to_numpy()

        if coords.shape[0] < num_points:
            raise ValueError(f"{file_name}: not enough points ({coords.shape[0]}) for sampling {num_points}")
        
        # ランダムサンプリング（numpyで）
        sampled_idx = np.random.choice(coords.shape[0], num_points, replace=False)
        pointcloud = coords[sampled_idx]

        # 正規化（各次元ごとに0~1）
        pointcloud = (pointcloud - pointcloud.min(axis=0)) / (pointcloud.max(axis=0) - pointcloud.min(axis=0))

        input_data_list.append(pointcloud)

    input_data = np.stack(input_data_list)  # Shape: (num_samples, num_points, 3)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    
    return DataLoader(TensorDataset(input_tensor), batch_size=1, shuffle=False)

def prepare_data(file_names, num_points=5000, device='cuda'):
    """
    Loads and preprocesses point cloud data from OFF files.
    Args:
        file_names (list of str): List of file paths to OFF files.
        num_points (int): Number of points to sample from each point cloud.
        device (str): Device to use ('cuda' or 'cpu').
    Returns:
        DataLoader: DataLoader object containing the preprocessed data.
    """
    input_data_list = []
    for file_name in file_names:
        with open(file_name, 'r') as f:
            verts, faces = read_off(f)
            pointcloud = PointSampler(num_points)((verts, faces))
            train_x, train_y, train_z = pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2]
            train_xn = (train_x - train_x.min()) / (train_x.max() - train_x.min())
            train_yn = (train_y - train_y.min()) / (train_y.max() - train_y.min())
            train_zn = (train_z - train_z.min()) / (train_z.max() - train_z.min())
            combined_data = np.concatenate((train_xn, train_yn, train_zn))
            input_data_list.append(combined_data)
    input_data = np.stack(input_data_list)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    return DataLoader(TensorDataset(input_tensor), batch_size=1, shuffle=False)

def prepare_data_for_ISAB(file_names, num_points=5000, device='cuda'):
    """
    Loads and preprocesses point cloud data for ISAB-based Set-VAE.
    Args:
        file_names (list of str): List of file paths to OFF files.
        num_points (int): Number of points to sample from each point cloud.
        device (str): Device to use ('cuda' or 'cpu').
    Returns:
        DataLoader: DataLoader object containing the preprocessed set-structured data.
    """
    input_data_list = []
    for file_name in file_names:
        with open(file_name, 'r') as f:
            verts, faces = read_off(f)
            pointcloud = PointSampler(num_points)((verts, faces))
            pointcloud = (pointcloud - pointcloud.min(axis=0)) / (pointcloud.max(axis=0) - pointcloud.min(axis=0))  # Normalize per dimension
            input_data_list.append(pointcloud)
    input_data = np.stack(input_data_list)  # Shape: (num_samples, num_points, 3)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    return DataLoader(TensorDataset(input_tensor), batch_size=1, shuffle=False)
