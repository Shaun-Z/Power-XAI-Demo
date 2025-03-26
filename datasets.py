import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class BinaryDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing binary voltage data.

    Args:
        folder_name (str): The name of the folder containing the voltage data CSV files.

    Attributes:
        data_df (pd.DataFrame): DataFrame containing the voltage data.
        data_outage_df (pd.DataFrame): DataFrame containing the voltage data during outages.
        n_buses (int): Number of voltage buses.
        time_steps (int): Number of time steps.
        data (torch.Tensor): Tensor containing the concatenated voltage data and outage data.
        labels (torch.Tensor): Tensor containing the labels for the data (0 for normal, 1 for outage).

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Returns the data and label at the specified index.
    """
    def __init__(self, folder_name):
        super().__init__()
        self.data_df = pd.read_csv(f'./data/{folder_name}/voltage_data.csv')
        self.data_outage_df = pd.read_csv(f'./data/{folder_name}/voltage_data_outage.csv')

        voltage_data = self.data_df.values
        voltage_data_outage = self.data_outage_df.values

        self.n_buses = voltage_data.shape[-1]  # Number of voltage buses
        self.time_steps = voltage_data.shape[0]  # Number of time steps

        self.data = torch.tensor(np.concatenate((voltage_data, voltage_data_outage), axis=0), dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(np.concatenate((np.zeros(voltage_data.shape[0]), np.ones(voltage_data_outage.shape[0])), axis=0)).unsqueeze(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]