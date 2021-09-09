import os
import numpy as np
import torch


feat_file = "./feats/dev/549-126410-0057.npy"
audio = np.load(feat_file)
inp = torch.from_numpy(audio).float()

print(type(inp), inp.shape)

