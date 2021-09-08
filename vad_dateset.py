import torch
import torch.utils.data as data


class VAD_Dataset(data.Dataset):
    def __init__(self, DB, loader, transform = None, *arg, **kwargs):
        self.DB = DB
        self.len = len(DB)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        feat_path = self.DB['filename'][index]

        feature, label = self.loader(feat_path)
        return feature, label

    def __len__(self):
        return self.len