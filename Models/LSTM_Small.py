import torch
import torch.nn as nn

class LSTMSmallNetwork(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, num_layers=2, bidirectional=True):
        super(LSTMSmallNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        # Adjust the input size for the fully connected layer based on bidirectional or not
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(in_features=fc_input_size, out_features=1)

    def forward(self, x):
        output, _status = self.lstm(x)
        output = output[:, -1, :]  # Take the output from the last time step
        output = self.fc1(torch.relu(output))
        return output