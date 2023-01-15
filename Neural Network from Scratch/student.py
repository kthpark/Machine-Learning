import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def one_hot(data: np.ndarray) -> np.ndarray:
    y = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y[rows, data] = 1
    return y


def plot(loss_history: list, accuracy_history: list, filename='plot'):
    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


def scale(x_train, x_test):
    X_max = np.max(x_train)
    x_train, x_test = x_train / X_max, x_test / X_max
    return x_train, x_test


def xavier(n_in, n_out):
    low, high = -np.sqrt(6 / (n_in + n_out)), np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(low, high, (n_in, n_out))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse(x, y):
    return np.mean((x - y) ** 2)


def mse_d(x, y):
    return 2 * (x - y)


class OneLayerNeural:
    def __init__(self, n_features, n_classes):
        self.W = xavier(n_features, n_classes)
        self.b = xavier(1, n_classes)

    def forward(self, X):
        return sigmoid(np.dot(X, self.W) + self.b)

    def backprop(self, X, y, alpha):
        error = (mse_d(self.forward(X), y) * sigmoid_d(np.dot(X, self.W) + self.b))

        dW = (np.dot(X.T, error)) / X.shape[0]
        db = np.mean(error, axis=0)

        self.W -= alpha * dW
        self.b -= alpha * db


def train(model_train, X, y, alpha, batch_size=100):
    n = X.shape[0]
    for i in range(0, n, batch_size):
        model_train.backprop(X[i:i + batch_size], y[i:i + batch_size], alpha)


def accuracy(model_train, X, y):
    y_pred = np.argmax(model_train.forward(X), axis=1)
    y_true = np.argmax(y, axis=1)
    return np.mean(y_pred == y_true)


class TwoLayerNeural:
    def __init__(self, n_features, n_classes):
        hidden_size = 64

        self.W = [xavier(n_features, hidden_size), xavier(hidden_size, n_classes)]
        self.b = [xavier(1, hidden_size), xavier(1, n_classes)]

    def forward(self, X):
        z = X
        for i in range(2):
            z = sigmoid(np.dot(z, self.W[i]) + self.b[i])
        return z

    def backprop(self, X, y, alpha):
        n = X.shape[0]
        biases = np.ones((1, n))
        yp = self.forward(X)

        loss_grad_1 = 2 * alpha / n * ((yp - y) * yp * (1 - yp))
        f1_out = sigmoid(np.dot(X, self.W[0]) + self.b[0])
        loss_grad_0 = np.dot(loss_grad_1, self.W[1].T) * f1_out * (1 - f1_out)

        self.W[0] -= np.dot(X.T, loss_grad_0)
        self.W[1] -= np.dot(f1_out.T, loss_grad_1)

        self.b[0] -= np.dot(biases, loss_grad_0)
        self.b[1] -= np.dot(biases, loss_grad_1)


if __name__ == '__main__':

    raw_train, raw_test = pd.read_csv('../Data/fashion-mnist_train.csv'), pd.read_csv('../Data/fashion-mnist_test.csv')
    X_train, X_test, y_train, y_test = raw_train[raw_train.columns[1:]].values, raw_test[raw_test.columns[1:]].values, \
        one_hot(raw_train['label'].values), one_hot(raw_test['label'].values)

    X_train, X_test = scale(X_train, X_test)

    n_features, n_classes = X_train.shape[1], y_train.shape[1]

    model = TwoLayerNeural(n_features, n_classes)

    acc_log_list = []

    for _ in range(20):
        train(model, X_train, y_train, 0.5)
        acc_log_list.append(accuracy(model, X_test, y_test))

    print(acc_log_list)
