import torch
from torch import nn

# LSTM for VAD
class VADnet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers):
        super(VADnet, self).__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_size, num_layers=num_layers, bias=True, batch_first=False,
                            bidirectional=True)
        self.fc = nn.Linear(in_features=2 * hidden_size,
                            out_features=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def init_hiddens(self, batch_size):
        # hidden state should be (num_layers*num_directions, batch_size, hidden_size)
        # returns a hidden state and a cell state
        return (torch.rand(size=(self.num_layers * 2, batch_size, self.hidden_size)),) * 2

    def forward(self, input_data, hiddens):
        '''
        input_data : (seq_len, batchsize, input_dim)
        '''
        outputs, hiddens = self.lstm(input_data, hiddens)
        # outputs: (seq_len, batch_size, num_directions* hidden_size)
        pred = self.fc(outputs)
        pred = self.sigmoid(pred)
        return pred