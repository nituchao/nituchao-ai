# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd

########################################################################
# 逻辑回归算法
########################################################################
# 模型: h = w_0*1 + w_1*x_1 + w_2*x_2 ... w_n*x_n
# 模型: h = np.sum(X, weights)
# 误差: error = y - h
# 权重更新: weights = weights + learning_rate * X.transpose() * error
########################################################################
# 输入特征矩阵X为(row, col), 模型构建过程中特征矩阵X为(row, col+1)。
# 将常数项bias，作为第一列放在输入特征矩阵X。
########################################################################
class LogisticRegression:
    
    def __init__(self):
        self.weights = []
        self.bias = 0
        return

    # sigmoid函数
    def __sigmoid(self, X):
        return 1.0 / (1 + np.exp(-X))

    # 初始化特征矩阵X
    def __init_X(self, X):
        # 将X格式转换成np.matrix
        X_matrix = np.mat(X.values)

        # 为X增加常数bias项列
        row, _ = X_matrix.shape
        bias_matrix = np.ones((row, 1))
        X_matrix_with_bias = np.c_[bias_matrix, X_matrix]

        return X_matrix_with_bias

    # 梯度上升求最优参数
    ########################################################################
    # X_in: 特征矩阵
    # y_in: 标签列
    # learning_rate: 学习率
    # num_iter: 最大迭代次数
    def __gradient_ascent(self, X, y, learning_rate=0.001, num_iter=500):
        _, col = np.shape(X)
        weights = np.ones((col, 1))
        
        for _ in range(num_iter):
            h = self.__sigmoid(np.sum(np.dot(X, weights)))
            error = (y - h)
            weights = weights + learning_rate * X.transpose() * error
        
        return weights

    # 随机梯度上升。
    # 当数据量比较大时，每次迭代都选择全量数据进行计算，计算量会非常大。
    # 所以采用每次迭代中一次只【依次选择】其中的一行数据进行更新权重。
    ########################################################################
    # X_in: 特征矩阵
    # y_in: 标签列
    # learning_rate: 学习率
    # num_iter: 最大迭代次数
    def __random_gradient_ascent0(self, X, y, learning_rate=0.001, num_iter=500):
        row, col = np.shape(X)
        weights = np.ones((col, 1))

        for _ in range(num_iter):
            for i in range(row):
                h = self.__sigmoid(np.sum(np.dot(X[i], weights)))
                error = y[i] - h
                weights = weights + learning_rate * X[i].transpose() * error
        
        return weights

    # 随机梯度上升。
    # 当数据量比较大时，每次迭代都选择全量数据进行计算，计算量会非常大。
    # 所以采用每次迭代中一次只【随机选择】其中的一行数据进行更新权重。
    ########################################################################
    # X_in: 特征矩阵
    # y_in: 标签列
    # learning_rate: 学习率
    # num_iter: 最大迭代次数
    def __random_gradient_ascent1(self, X, y, learning_rate=0.001, num_iter=500):
        row, col = np.shape(X)
        weights = np.ones((col, 1))

        for j in range(num_iter):
            dataIndex = [i for i in range(row)]
            for i in range(row):
                alpha = 4 / (1 + j + i) + learning_rate
                randIndex = int(np.random.uniform(0, len(dataIndex)))
                h = self.__sigmoid(np.sum(X[randIndex] * weights))
                error = y[randIndex] - h
                weights = weights + alpha * X[randIndex].transpose() * error

                del(dataIndex[randIndex])
        
        return weights

    # 模型训练
    ########################################################################
    # X: 特征矩阵
    # y: 标签列
    # learning_rate: 学习率
    # num_iter: 最大迭代次数
    # optimization: 迭代优化算法，默认值1。可选值：1: gradient_ascent;2: random_gradient_ascent0;3: random_gradient_ascent1;
    def fit(self, X, y, learning_rate=0.001, num_iter=500, optimization=1) -> None:
        # 将X格式转换为np.matrix
        X_matrix = self.__init_X(X)
        # 将y格式转换为np.matrix
        y_matrix = np.mat(y.values.reshape(-1, 1))

        # 根据选择的优化器进行迭代
        if optimization == 2:
            weights = self.__random_gradient_ascent0(X_matrix, y_matrix, learning_rate, num_iter)
        elif optimization == 3:
            weights = self.__random_gradient_ascent1(X_matrix, y_matrix, learning_rate, num_iter)
        else:
            weights = self.__gradient_ascent(X_matrix, y_matrix, learning_rate, num_iter)

        # 设置成员变量weights
        row, _ = weights.shape
        for i in range(row):
            if i == 0:
                self.bias = weights.item(i, 0)
            else :
                self.weights.append(weights.item(i, 0)) 

        return

    # 模型预测
    ########################################################################
    # X: 特征矩阵
    def predict(self, X) -> float:
        # 将X格式转换为np.matrix
        X_matrix = self.__init_X(X)

        # 获取模型训练结果权重
        weights = [self.bias] + self.weights

        # 带入参数计算模型预测结果
        y_predict = self.__sigmoid(np.sum(np.dot(X_matrix, weights)))

        return y_predict

