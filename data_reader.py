import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime



class DataReader:
    def __init__(self):
        self.LOOKBACK = 20
        self.NUM_BANDS = 125
        self.raw_data = pd.read_csv("data/moisture.csv")
        self.data = torch.zeros(len(self.raw_data), 2)
        self.bands = torch.zeros(len(self.raw_data), self.NUM_BANDS)
        for index, row in self.raw_data.iterrows():
            timestamp = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S").timestamp()
            moisture = row[2]
            temperature = row[3]
            self.data[index,0] = moisture
            self.data[index,1] = temperature
            self.bands[index] = torch.tensor(row[4:])

        self.moisture_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.temperature_scaler = MinMaxScaler(feature_range=(-1, 1))

        temporary_moisture = self.data[:,0].reshape(-1,1)
        temporary_moisture = self.moisture_scaler.fit_transform(temporary_moisture)
        self.data[:,0] = torch.tensor(temporary_moisture[:,0])

        temporary_temperature = self.data[:,1].reshape(-1,1)
        temporary_temperature = self.temperature_scaler.fit_transform(temporary_temperature)
        self.data[:,1] = torch.tensor(temporary_temperature[:,0])

        self.data_size = len(self.bands) - self.LOOKBACK
        sequences = torch.zeros(self.data_size, self.LOOKBACK-1, 125)
        results = torch.zeros(self.data_size, 1)

        for index in range(self.data_size):
            sequences[index,:] = self.bands[index : index + self.LOOKBACK - 1]
            results[index,0] = self.data[index + self.LOOKBACK -1, 0]

        test_set_size = int(0.3 * self.data_size)
        train_set_size = self.data_size - test_set_size

        self.x_train = sequences[0:train_set_size]
        self.y_train = results[0:train_set_size]

        self.x_test = sequences[train_set_size:train_set_size + test_set_size]
        self.y_test = results[train_set_size:train_set_size + test_set_size]

    def get_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test


if __name__ == "__main__":
    dr = DataReader()
    x_train, y_train, x_test, y_test = dr.get_data()

