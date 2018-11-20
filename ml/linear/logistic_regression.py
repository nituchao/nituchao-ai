# -*- coding: UTF-8 -*-

import numpy as np

class LogisticRegression:
    
    def __init__(self):
        return

    # sigmoid函数
    def sigmoid(self, X):
        return 1.0 / (1 + np.exp(-X))

    # 梯度上升求最优参数
    def gradAscent(self, dataMat, lableMat):
        dataMatrix = np.mat(dataMat)
        classLabels = np.mat(lableMat)

        m, n = np.shape(dataMatrix)
        alpha = 0.001
        maxCycles = 500
        weights = np.ones((n, 1))
        
        for _ in range(maxCycles):
            h = self.sigmoid(np.sum(np.dot(dataMatrix, weights)))
            error = (classLabels - h)
            weights = weights + alpha * dataMatrix.transpose() * error.transpose()
        
        return weights

    # 随机梯度上升。
    # 当数据量比较大时，每次迭代都选择全量数据进行计算，计算量会非常大。
    # 所以采用每次迭代中一次只选择其中的一行数据进行更新权重。
    def stocGradAscent0(self, dataMat, lableMat):
        dataMatrix = np.mat(dataMat)
        classLabels = lableMat
        
        m, n = np.shape(dataMatrix)
        alpha = 0.001
        maxCycles = 500
        weights = np.ones((n, 1))

        for k in range(maxCycles):
            for i in range(m):
                h = self.sigmoid(sum(np.dot(dataMatrix[i], weights)))
                error = classLabels[i] - h
                weights = weights + alpha * dataMatrix[i].transpose() * error
        
        return weights

    def stocGradAscent1(self, dataMat, labelMat):
        dataMatrix = np.mat(dataMat)
        classLabels = labelMat

        m, n = np.shape(dataMatrix)
        weights = np.ones((n, 1))
        maxCycles = 500

        for j in range(maxCycles):
            dataIndex = [i for i in range(m)]
            for i in range(m):
                alpha = 4 / (1 + j + i) + 0.0001
                randIndex = int(np.random.uniform(0, len(dataIndex)))
                h = self.sigmoid(np.sum(dataMatrix[randIndex] * weights))
                error = classLabels[randIndex] - h
                weights = weights + alpha * dataMatrix[randIndex].transpose() * error

                del(dataIndex[randIndex])
        
        return weights