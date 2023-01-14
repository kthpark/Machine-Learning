from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = np.array([])
        self.intercept = float

    def fit(self, X, y):
        X0 = X.copy()
        if self.fit_intercept:
            I = pd.Series(1, index=X0.index)
            X0.insert(loc=0, column="I", value=I)
        Xn = X0.to_numpy()
        y = y.to_numpy()
        Xt = Xn.T
        beta = np.linalg.inv(Xt @ Xn) @ Xt @ y
        self.coefficient = beta
        if self.fit_intercept:
            self.intercept = beta[0]
            self.coefficient = beta[1:]

    def predict(self, X):
        X = X.to_numpy()
        y = self.intercept + X @ self.coefficient
        return y

    @staticmethod
    def rmse(y, yhat):
        n = y.shape[0]
        return np.sqrt(np.sum((y - yhat) ** 2) / n)

    @staticmethod
    def r2_score(y, yhat):
        return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


if __name__ == '__main__':
    df = pd.read_csv('linregdata.csv')
    custom_model = CustomLinearRegression(fit_intercept=True)
    model = LinearRegression(fit_intercept=True)

    X_train, y_train = df[['f1', 'f2', 'f3']], df['y']

    custom_model.fit(X_train, y_train)
    y_pred = custom_model.predict(X_train)
    rmse = custom_model.rmse(y_train, y_pred)
    r2 = custom_model.r2_score(y_train, y_pred)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    rmse_model = np.sqrt(mean_squared_error(y_train, y_pred))
    r2_model = r2_score(y_train, y_pred)

    output = {'Intercept': custom_model.intercept - model.intercept_, 'Coefficient': custom_model.coefficient - model.coef_,
              'R2': r2 - r2_model, 'RMSE': rmse - rmse_model}
    print(output)
