# -*- coding: UTF-8 -*-

import numpy as np

class LinearRegression:

    def __init__(self):
        return

    # 计算均方误差
    def compute_error(self, X, y, w, b):
        total_error = 0
        row = float(X.shape[0])

        # 使用矩阵运算，计算均方误差
        total_error = np.square(y - np.dot(X, w) - b)
        total_error = np.sum(total_error, axis=0)

        return total_error/row

    # 计算梯度
    def compute_gradient(self, X, y, w_current, b_current, learning_rate):
        b_gradient = 0
        w_gradient = 0

        row = float(X.shape[0])

        # 通过矩阵运算，计算梯度
        b_gradient = -(2 / row) * (y - np.dot(X, w_current) - b_current)
        b_gradient = np.sum(b_gradient, axis=0)

        w_gradient = -(2 / row) * np.dot(X.T, (y - np.dot(X, w_current) - b_current))
        w_gradient = np.sum(w_gradient, axis=0)

        # 更新b和m
        new_b = b_current - (learning_rate * b_gradient)
        new_w = w_current - (learning_rate * w_gradient)

        return [new_b, new_w]

    # 迭代优化器
    def optimizer(self, X, y, starting_w, starting_b, learning_rate, num_iter):
        b = starting_b
        w = starting_w

        # 使用梯度下降进行迭代
        for i in range(num_iter):
            # update b and m with the new more accurate b and m by preforming
            # the gradient setup
            b, w = self.compute_gradient(X, y, w, b, learning_rate)
            
            if i % 100 == 0:
                print("iter {0}: error = {1}".format(i, self.compute_error(X, y, w, b)))

        return [b, w]

    # 训练模型
    def fit(self, X, y, learning_rate=0.0001, max_num_iter=100):
        X_np = X.values
        y_np = y.values.reshape(-1, 1)

        row = X_np.shape[0]
        col = X_np.shape[1]

        initial_b = np.array(np.zeros(row)).reshape(row, -1)
        initial_w = np.array(np.zeros(col)).reshape(col, -1)

        print("=======")
        print('X{0} * w{1} + b{2} = y{3}'.format(type(X_np), type(initial_w), type(initial_b), type(y_np)))
        print('X{0} * w{1} + b{2} = y{3}'.format(X_np.shape, initial_w.shape, initial_b.shape, y_np.shape))
        print("=======")

        # 训练模型
        # print('initial variables:\n initial_b = {0}\n inital_w = {1}\n error of begin = {2} \n'.format(initial_b.T, initial_w.T, self.compute_error(X_np, y_np, initial_w, initial_b)))

        # 优化参数
        [b, w] = self.optimizer(X_np, y_np, initial_w, initial_b, learning_rate, max_num_iter)

        # 打印最终的b,m,error
        print('final formula parmaters:\n num_iter = {0}\n b = {1}\n w = {2}\n error of end = {3} \n'.format(max_num_iter, b, w, self.compute_error(X_np, y_np, w, b)))
