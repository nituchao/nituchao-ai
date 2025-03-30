# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from ml.linear.linear_regression import LinearRegression
from ml.linear.logistic_regression import LogisticRegression

def main():
    # 加载数据
    data = pd.read_csv('data/data2.csv', delimiter=',')
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    max_num_iter = 1000
    learning_rate = 0.00001

    print(X.shape)
    print(y.shape)

    lr = LinearRegression()
    lr.fit(X, y, learning_rate, max_num_iter)

def main2():
         #读取数据（这里只有两个特征）
    filename = "/Users/liang/Documents/workspace/python/nituchao-deeplearning/data/data_lr.txt"
    data = pd.read_csv(filename, delimiter=',', header=None)

    dataMat = data.iloc[:, 0:2]
    labelMat = data.iloc[:, 2]

    lr = LogisticRegression()
    lr.fit(dataMat, labelMat, optimization=1)
    weights = lr.weights
    bias = lr.bias

    print('z = {} + {}*x + {}*y'.format(bias, weights[0], weights[1]))

    for i in range(len(labelMat)):
        X = dataMat.iloc[i, :]
        y = labelMat.values[i]
        y_predict = lr.predict(X)
        print('x is {}, y is {}, predict_y is {}.'.format(X.values, y, y_predict))

if __name__ == "__main__":
    main2()