import os
import torch
from torch.utils.data import Dataset, DataLoader
from enformer_pytorch.data import seq_padding, str_to_one_hot, str_to_seq_indices
import pandas as pd

class mouse_8_25(Dataset):
    def __init__(self, cell_type='mixed_meso', data_class='train'):
        """
        Args:
            cell_type (string): Cell type in mouse_8_25 embryonic stage
            data_class (string): train/test/val
        """
        # Load the CSV file
        self.base_dir = '/g/data/zk16/zelun/z_li_hon/DNABERT_2/data_prep/'
        # NOTE: not spliting on comma here
        self.data = pd.read_csv(os.path.join(self.base_dir, cell_type, f"{data_class}.csv"))
        self.num_workers = 96

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Depending on the structure of your CSV, 
        # extract the data and return it as a torch.Tensor
        # TODO: Add in the fasta extractor here
        sequence = self.data.iloc[idx, 0]  # for example, if sequences are in the second column
        target = self.data.iloc[idx, 1]
        sequence_tensor = str_to_seq_indices(seq_padding(sequence))
        # NOTE: Testing with randn tensor need to change it
        target_tensor = torch.tensor(target)
        # target_tensor = torch.randn(1, 200, 128)
        # NOTE: CHANGE THE RETURN
        # print(f"==== target is {target_tensor} ====")
        # print(f"seq SHAPE {sequence_tensor.shape} target SHAPE {target_tensor.shape}")
        return sequence_tensor, target_tensor
        # return {'sequence': sequence_tensor, 'target': target_tensor}


