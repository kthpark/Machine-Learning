import os
import sys
import requests

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape


def stage1(data):
    X, y = pd.DataFrame(data['rating']), data['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(round(model.intercept_, 5), round(model.coef_[0], 5), round(mape(y_test, model.predict(X_test)), 5))


def mape_power(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    model = LinearRegression()
    model.fit(X_train, y_train)
    mape_pow = round(mape(y_test, model.predict(X_test)), 5)
    return mape_pow


def stage2(data):
    X, y = pd.DataFrame(data['rating']), data['salary']
    X2, X3, X4 = X.pow(2), X.pow(3), X.pow(4)
    mape2, mape3, mape4 = mape_power(X2, y), mape_power(X3, y), mape_power(X4, y)

    print(min(mape2, mape3, mape4))


def stage3(data):
    X, y = data.loc[:, data.columns != 'salary'], data['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    model = LinearRegression()
    model.fit(X_train, y_train)
    model_coef = list()
    for i in model.coef_:
        model_coef.append(round(float(i), 5))
    print(model_coef)


def calc_reg(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    model = LinearRegression()
    model.fit(X_train, y_train)
    test_prediction = model.predict(X_test)
    mape_test = mape(y_test, test_prediction)
    return mape_test


def stage4(data):
    X, y = data.loc[:, data.columns != 'salary'], data['salary']
    var_list = ['rating', 'age', 'experience']
    m_min = sys.maxsize

    for i in range(3):
        X1 = X.drop(columns=var_list[i])
        m = calc_reg(X1, y)
        if m < m_min:
            m_min = m
        X1 = X.drop(columns=[var_list[(i + 1) % 3], var_list[(i + 2) % 3]])
        m = calc_reg(X1, y)
        if m < m_min:
            m_min = m

    print(round(m_min, 5))


def calc_reg2(X, y, y_val):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    model = LinearRegression()
    model.fit(X_train, y_train)
    test_prediction = model.predict(X_test)
    if y_val != 0:
        y_val = y_train.median()
    test_prediction = np.where(test_prediction < 0, y_val, test_prediction)
    mape_test = mape(y_test, test_prediction)
    return mape_test


def stage5(data):
    X, y = pd.DataFrame(data).drop(columns=['salary', 'age', 'experience']), data['salary']
    m_0 = calc_reg2(X, y, 0)
    m_1 = calc_reg2(X, y, 1)

    print(round(min(m_0, m_1), 5))


def main():
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    if 'data.csv' not in os.listdir('../Data'):
        url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/data.csv', 'wb').write(r.content)

    data = pd.read_csv('../Data/data.csv')
    # stage1(data)
    # stage2(data)
    # stage3(data)
    # stage4(data)
    stage5(data)


if __name__ == '__main__':
    main()
