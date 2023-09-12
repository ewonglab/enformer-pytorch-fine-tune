import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class mouse_8_25(Dataset):
    def __init__(self, cell_type='allantois'):
        """
        Args:
            cell_type (string): Cell type in mouse_8_25 embryonic stage
        """
        # Load the CSV file
        self.base_dir = '/g/data/zk16/zelun/z_li_hon/DNABERT_2/data_prep/'
        # NOTE: not spliting on comma here
        self.train_data = pd.read_csv(os.path.join(self.base_dir, cell_type, 'train.csv'))
        self.val_data = pd.read_csv(os.path.join(self.base_dir, cell_type, 'val.csv'))

    def __len__(self):
        return (len(self.train_data), len(self.val_data))

    def __getitem__(self, idx):
        # Depending on the structure of your CSV, 
        # extract the data and return it as a torch.Tensor
        # TODO: Add in the fasta extractor here
        sequence = self.data.iloc[idx, 1]  # for example, if sequences are in the second column
        sequence_tensor = torch.tensor([float(x) for x in sequence.split(',')])  # assuming sequences are comma-separated in the CSV
        return sequence_tensor


