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
    filename = "/Users/liang/Documents/workspace/python/nituchao-deeplearning/data/data_lr.csv"
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]), float(lineArr[1])])   #前面的1，表示方程的常量。比如两个特征X1,X2，共需要三个参数，W1+W2*X1+W3*X2
        labelMat.append(int(lineArr[2]))

    lr = LogisticRegression()
    lr.fit(dataMat, labelMat, optimization=1)
    weights = lr.weights
    bias = lr.bias

    print('z = {} + {}*x + {}*y'.format(bias, weights[0], weights[1]))

    X = dataMat[1]
    y = labelMat[1]
    y_predict = lr.predict(X)
    print('x is {}, y is {}, predict_y is {}.'.format(X, y, y_predict))

if __name__ == "__main__":
    main2()