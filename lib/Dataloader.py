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

def read_stl_text(file):
    """
    Reads an ASCII STL file and extracts vertex coordinates and face indices.

    Args:
        file (file object): A file object opened in text mode containing an ASCII STL file.

    Returns:
        tuple: A tuple containing:
            - vertices (list of tuples): A list of unique vertex coordinates (x, y, z) as float tuples.
            - faces (list of lists): A list of faces, each represented as a list of three vertex indices.
    """
    verts = []  # List to store unique vertex coordinates
    faces = []  # List to store faces (triangles as indices)
    vert_map = {}  # Dictionary to map vertex coordinates to their index
    current_face = []  # Temporary list to store the current face's vertices

    for line in file:
        parts = line.strip().split()
        if not parts:
            continue
        
        if parts[0] == 'vertex':
            vertex = tuple(float(p) for p in parts[1:])
            if vertex not in vert_map:
                vert_map[vertex] = len(verts)
                verts.append(vertex)
            current_face.append(vert_map[vertex])
            
            if len(current_face) == 3:  # A face consists of exactly 3 vertices
                faces.append(current_face)
                current_face = []  # Reset for the next face

    return verts, faces

def prepare_data_from_stl(file_names, num_points=5000, device='cuda'):
    """
    Loads and preprocesses point cloud data from ASCII STL files.
    Extracts vertices from STL faces, normalizes them, and returns a DataLoader.

    Args:
        file_names (list of str): List of ASCII STL file paths.
        num_points (int): Number of points to sample from each file.
        device (str): 'cuda' or 'cpu'.

    Returns:
        DataLoader: DataLoader containing normalized point cloud tensors.
    """
    input_data_list = []

    for file_name in file_names:
        with open(file_name, 'r') as f:
            verts, faces = read_stl_text(f)
            verts_array = np.array(verts)

            if verts_array.shape[0] < num_points:
                raise ValueError(f"{file_name}: not enough vertices ({verts_array.shape[0]}) for sampling {num_points}")

            sampled_idx = np.random.choice(verts_array.shape[0], num_points, replace=False)
            pointcloud = verts_array[sampled_idx]

            # Normalize each dimension to [0, 1]
            pointcloud = (pointcloud - pointcloud.min(axis=0)) / (pointcloud.max(axis=0) - pointcloud.min(axis=0))

            input_data_list.append(pointcloud)

    input_data = np.stack(input_data_list)  # (N_samples, N_points, 3)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

    return DataLoader(TensorDataset(input_tensor), batch_size=1, shuffle=False)
