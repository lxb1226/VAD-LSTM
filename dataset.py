from torch.utils import data
import numpy as np
import os


class DatasetCSV(data.Dataset):
    """
    Dataset using Txt files
    """

    def __init__(self, data_root, csv_file, is_train):
        self.data = []
        self.is_train = is_train

        with open(csv_file, 'r') as f:
            lines = f.read().split('\n')
            for line in lines:
                if len(line):
                    feat_file = line[0]
                    feat = np.load(os.path.join(data_root, feat_file))
                    self.data.append([feat])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
