import torch
import torch.nn as nn
from machine import Machine
import time
import numpy as np
import data_reader
import os
import train
import plotter
from data_reader import DataReader


def test():
    model = Machine()
    #if not os.path.isfile("models/machine.h5"):
    train.train()
    model = torch.load("models/machine.h5")

    dr = DataReader()
    image_test, moisture_test, temperature_test, y_test = dr.get_test_data()

    criterion = torch.nn.MSELoss(reduction='mean')
    y_test_pred = model(image_test, moisture_test, temperature_test)
    loss = criterion(y_test_pred, y_test)
    print(f"Loss: {loss}")
    plotter.plot(y_test.detach().numpy(), y_test_pred.detach().numpy())


if __name__ == "__main__":
    test()
