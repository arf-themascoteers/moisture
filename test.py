import torch
import torch.nn as nn
from machine import Machine
import time
import numpy as np
import data_reader
import os
import train
import plotter

def test():
    model = Machine()
    if not os.path.isfile("models/machine.h5"):
        train.train()
    model = torch.load("models/machine.h5")

    x_train, y_train, x_test, y_test = data_reader.get_data()

    criterion = torch.nn.MSELoss(reduction='mean')
    y_test_pred = model(x_test)
    loss = criterion(y_test_pred, y_test)
    print(f"Loss: {loss}")
    y_test = y_test[:,0]
    y_test_pred = y_test_pred[:,0]
    plotter.plot(y_test.detach().numpy(), y_test_pred.detach().numpy())


if __name__ == "__main__":
    test()
