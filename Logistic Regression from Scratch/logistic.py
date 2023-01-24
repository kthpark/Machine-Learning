import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.coefficients = None
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.k = 0

    @staticmethod
    def sigmoid(t):
        return 1 / (1 + np.exp(-t))

    def predict_proba(self, row, coefficients):
        t = coefficients[0] + np.dot(row, coefficients[1:]) if self.fit_intercept else np.dot(row, coefficients)
        return self.sigmoid(t)

    def check_coefficients(self):
        self.coefficients = [0. for _ in range(X_train.shape[1])]
        if self.fit_intercept:
            self.coefficients = [0.] + self.coefficients
            self.k = 1
        else:
            self.k = 0

    def fit_mse(self, X_train, y_train):
        self.check_coefficients()

        for x in range(self.n_epoch):
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coefficients)
                delta = self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat)
                self.coefficients[0] = self.coefficients[0] - delta

                for j in range(len(row)):
                    self.coefficients[j + self.k] = self.coefficients[j + self.k] - delta * row[j]

                if x in [0, self.n_epoch - 1]:
                    y_hat = self.predict_proba(row, self.coefficients)
                    mse_err = np.power(y_hat - y_train[i], 2) / X_train.shape[0]
                    mse_error_first.append(mse_err) if x == 0 else mse_error_last.append(mse_err)

    def fit_log_los(self, X_train, y_train):
        self.check_coefficients()

        for x in range(self.n_epoch):
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coefficients)
                delta = self.l_rate * (y_hat - y_train[i]) / X_train.shape[0]
                self.coefficients[0] = self.coefficients[0] - delta

                for j in range(len(row)):
                    self.coefficients[j + self.k] = self.coefficients[j + self.k] - delta * row[j]

                if x in [0, self.n_epoch - 1]:
                    y_hat = self.predict_proba(row, self.coefficients)
                    log_err = -(y_train[i] * np.log(y_hat) + (1 - y_train[i]) * np.log(1 - y_hat)) / X_train.shape[0]
                    log_loss_error_first.append(log_err) if x == 0 else log_loss_error_last.append(log_err)

    def predict(self, sample, cut_off=0.5):
        return [int(self.predict_proba(row, self.coefficients) >= cut_off) for row in sample]


if __name__ == '__main__':
    df = load_breast_cancer(as_frame=True).frame
    X, y = StandardScaler().fit_transform(df[['worst concave points', 'worst perimeter', 'worst radius']]), df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)

    acc_score = accuracy_score(y_test, LogisticRegression(fit_intercept=True).fit(X_train, y_train).predict(X_test))

    Regression = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
    mse_error_first, mse_error_last, log_loss_error_first, log_loss_error_last = [], [], [], []

    Regression.fit_mse(X_train, y_train.to_numpy())
    acc_mse = accuracy_score(y_test, Regression.predict(X_test))
    Regression.fit_log_los(X_train, y_train.to_numpy())
    acc_log = accuracy_score(y_test, Regression.predict(X_test))

    properties = {'mse_accuracy': acc_mse,
                  'log_loss_accuracy': acc_log,
                  'sklearn_accuracy': acc_score,
                  'mse_error_first': mse_error_first,
                  'mse_error_last': mse_error_last,
                  'log_loss_error_first': log_loss_error_first,
                  'log_loss_error_last': log_loss_error_last}

    print(properties)
