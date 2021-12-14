import torch
import torch.nn as nn

class Machine(nn.Module):
    def __init__(self):
        super(Machine, self).__init__()
        self.hidden_dim = 8
        self.num_layers = 1

        self.image_lstm = nn.LSTM(125, self.hidden_dim, self.num_layers, batch_first=True)
        self.moisture_lstm = nn.LSTM(1, self.hidden_dim, self.num_layers, batch_first=True)
        self.temperature_lstm = nn.LSTM(1, self.hidden_dim, self.num_layers, batch_first=True)

        self.image_fc = nn.Linear(self.hidden_dim, 1)
        self.moisture_fc = nn.Linear(self.hidden_dim, 1)
        self.temperature_fc = nn.Linear(self.hidden_dim, 1)

        self.last_fc = nn.Linear(3,1)

    def forward(self, image, moisture, temperature):
        h0_image = torch.zeros(self.num_layers, image.size(0), self.hidden_dim).requires_grad_()
        c0_image = torch.zeros(self.num_layers, image.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.image_lstm(image, (h0_image.detach(), c0_image.detach()))
        image_out = self.image_fc(out[:,-1,:])

        h0_moisture = torch.zeros(self.num_layers, moisture.size(0), self.hidden_dim).requires_grad_()
        c0_moisture = torch.zeros(self.num_layers, moisture.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.moisture_lstm(moisture, (h0_moisture.detach(), c0_moisture.detach()))
        moisture_out = self.moisture_fc(out[:,-1,:])

        h0_temperature = torch.zeros(self.num_layers, temperature.size(0), self.hidden_dim).requires_grad_()
        c0_temperature = torch.zeros(self.num_layers, temperature.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.moisture_lstm(temperature, (h0_temperature.detach(), c0_temperature.detach()))
        temperature_out = self.temperature_fc(out[:,-1,:])

        x = torch.cat((image_out, moisture_out, temperature_out), dim=1)
        x = self.last_fc(x)

        return x