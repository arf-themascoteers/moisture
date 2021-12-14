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
        self.moisture = torch.zeros(len(self.raw_data))
        self.temperature = torch.zeros(len(self.raw_data))
        self.bands = torch.zeros(len(self.raw_data), self.NUM_BANDS)
        for index, row in self.raw_data.iterrows():
            #timestamp = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S").timestamp()
            self.moisture[index] = row[2]
            self.temperature[index] = row[3]
            self.bands[index] = torch.tensor(row[4:])

        self.moisture_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.temperature_scaler = MinMaxScaler(feature_range=(-1, 1))

        temporary_moisture = self.moisture.reshape(-1,1)
        temporary_moisture = self.moisture_scaler.fit_transform(temporary_moisture)
        self.moisture = torch.tensor(temporary_moisture[:,0])

        temporary_temperature = self.temperature.reshape(-1,1)
        temporary_temperature = self.temperature_scaler.fit_transform(temporary_temperature)
        self.temperature = torch.tensor(temporary_temperature[:,0])

        self.data_size = len(self.bands) - self.LOOKBACK
        image_sequences = torch.zeros(self.data_size, self.LOOKBACK-1, 125)
        moisture_sequences = torch.zeros(self.data_size, self.LOOKBACK-1, 1)
        temperature_sequences = torch.zeros(self.data_size, self.LOOKBACK-1, 1)
        results = torch.zeros(self.data_size, 1)

        for index in range(self.data_size):
            image_sequences[index,:] = self.bands[index : index + self.LOOKBACK - 1]
            moisture_sequences[index, :, 0] = self.moisture[index : index + self.LOOKBACK - 1]
            temperature_sequences[index, :, 0] = self.temperature[index : index + self.LOOKBACK - 1]
            results[index,0] = self.moisture[index + self.LOOKBACK - 1]

        test_set_size = int(0.3 * self.data_size)
        train_set_size = self.data_size - test_set_size

        self.image_train = image_sequences[0:train_set_size]
        self.moisture_train = moisture_sequences[0:train_set_size]
        self.temperature_train = temperature_sequences[0:train_set_size]
        self.y_train = results[0:train_set_size]

        self.image_test = image_sequences[train_set_size:train_set_size + test_set_size]
        self.moisture_test = moisture_sequences[train_set_size:train_set_size + test_set_size]
        self.temperature_test = temperature_sequences[train_set_size:train_set_size + test_set_size]
        self.y_test = results[train_set_size:train_set_size + test_set_size]

    def get_train_data(self):
        return self.image_train, self.moisture_train, self.temperature_train, self.y_train

    def get_test_data(self):
        return self.image_test, self.moisture_test, self.temperature_test, self.y_test



