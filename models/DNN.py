import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function


class DNN_VAD(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=1):
        super(DNN_VAD, self).__init__()
        # input: (seq_len * 1 * input_dim) output: (sql_len * 1 * hidden_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        # input: (sql_len * 1 * hidden_size) output: (sql_len * 1 * hidden_size)
        self.bn1 = nn.BatchNorm1d(1)
        self.fc1_drop = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(1)
        self.fc2_drop = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(1)
        self.fc3_drop = nn.Dropout(p=0.2)

        self.last = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (seq_len, batch_size(1), input_dim)

        out = F.relu(self.bn1((self.fc1(x))))
        out = F.relu(self.bn2((self.fc2(out))))
        out = F.relu(self.bn3((self.fc3(out))))

        out = self.last(out)
        out = self.sigmoid(out)
        return out
