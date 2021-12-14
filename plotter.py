import matplotlib.pyplot as plt
import data_reader

def plot(y, y_pred):
    plt.plot(y, label="Original")
    plt.plot(y_pred, 'r', label="Prediction" )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    _,_,x_test, y_test = data_reader.get_data()
    y_test = y_test[:,0]
    y_test2 = y_test + 0.01
    plot(y_test.numpy(), y_test2.numpy())


