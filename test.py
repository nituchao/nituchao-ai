# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from ml.linear.linear_regression import LinearRegression

def main():
    # 加载数据
    data = pd.read_csv('/Users/liang/Documents/workspace/python/nituchao-deeplearning/ml/data/data2.csv', delimiter=',')
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    max_num_iter = 1000
    learning_rate = 0.00001

    print(X.shape)
    print(y.shape)

    lr = LinearRegression()
    lr.fit(X, y, learning_rate, max_num_iter)

if __name__ == "__main__":
    main()