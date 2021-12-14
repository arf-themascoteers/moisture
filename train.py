import torch
import torch.nn as nn
from machine import Machine
import time
import numpy as np
import data_reader

def train():

    model = Machine()
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    x_train, y_train, x_test, y_test = data_reader.get_data()

    start_time = time.time()

    for t in range(500):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        print("Epoch ", t, "MSE: ", loss.item())
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    training_time = time.time() - start_time
    print("Training time: {}".format(training_time))
    torch.save(model, "models/machine.h5")


if __name__ == "__main__":
    train()
