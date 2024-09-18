import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch

dataset_path = "/home/bilz/datasets/qead/q/"
dataset_name = "Bearing.csv"

def load_datasets(dataset_filename):
    print("Loading Bearing dataset...")
    try:
        file_path = dataset_filename
    except:
        file_path = os.path.join(dataset_path, dataset_name)
    data_Bearing = pd.read_csv(file_path)
    print(f"Loaded Bearing dataset with columns: {data_Bearing.columns}")
    return {'Bearing': data_Bearing}

def preprocess_data(data, qubit_no=20):
    print(f"Preprocessing bearing data with qubit no {qubit_no}...")

    if data.empty:
        raise ValueError("Dataset is empty.")

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    if 'Bearing 1' in data.columns:
        data['Bearing 1'] = scaler.fit_transform(data['Bearing 1'].values.reshape(-1, 1))
        X = np.array([data['Bearing 1'].values[i:i + qubit_no] for i in range(len(data) - qubit_no)])
        y_true = np.array([1 if val > 0.8 else 0 for val in data['Bearing 1'][qubit_no:]])
    else:
        raise ValueError("Unknown data format in dataset.")

    print("Preprocessing complete.")
    return X, y_true

# Custom Dataset class for PyTorch
class BearingDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
