import matplotlib.pyplot as plt
import data_reader
from data_reader import DataReader


def plot(y, y_pred):
    plt.plot(y, label="Original")
    plt.plot(y_pred, 'r', label="Prediction" )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    _,y_train,_, y_test = DataReader().get_data()
    #y_test2 = y_test + 0.01
    y_train2 = y_train + 0.01
    y_test2 = y_test + 0.01
    plot(y_test.numpy(), y_test2.numpy())
    #plot(y_train.numpy(), y_train2.numpy())


