import torch
import torch.nn as nn

class Machine(nn.Module):
    def __init__(self):
        super(Machine, self).__init__()
        self.hidden_dim = 16
        self.num_layers = 1

        self.lstm = nn.LSTM(1, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, 1)
        self.lrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc1(out[:,-1,:])
        # out = self.lrelu(out)
        # out = self.fc2(out)
        return out