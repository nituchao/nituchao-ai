import numpy as np
import pandas as pd
import cvxopt

class SVM(object):

    def __init__(self):
        self.C = None
        self.kernel = None

        self.sv_x = None
        self.sv_y = None
        self.sv_index = None

        self.bias = None
        self.alphas = None
        self.weights = None
        return

    def fit(self, X, y, optimize='soft', kernel='polynomial_kernel', C=1.0):
        """
        API for train svm model

        :param X: The feature matrix
        :param y: The lable matrix
        :param optimize: optimize algorithm, one of [hard, *soft, smo]
        :param kernel: kernel convertion, one of [linear_kernel, *polynomial_kernel, gaussian_kernel]
        :param c: penalty parameter, default 1.0
        :return: returns nothing
        """
        self.kernel = kernel
        self.C = C

        X_ = X.astype(float)
        y_ = y.astype(float)

        # chose optimization
        if optimize == 'soft':
            self.__optimize_soft_interval_by_qp(X_, y_, kernel=kernel, C=C)
        elif optimize == 'hard':
            self.__optimize_hard_interval_by_qp(X_, y_)
        elif optimize == 'smo':
            print("Use smo to solve svm")
        else:
            print("Error: param 'optimize' must one of [hard,soft], which is {}".format(optimize))

    def predict(self, X):
        X_ = X.astype(float)

        return np.sign(self.predict_probability(X_))

    def predict_probability(self, X):
        X_ = X.astype(float)

        if self.weights is not None:
            return np.dot(X_, self.weights) + self.bias
        else:
            y_predict = np.zeros(len(X_))
            for i in range(len(X_)):
                sum = 0
                for alpha, sv_y, sv_x in zip(self.alphas, self.sv_y, self.sv_x):
                    sum += alpha * sv_y * self.__kernel_convert(X[i], sv_x, kernel=self.kernel)
                y_predict[i] = sum
            return y_predict + self.bias 

    def __optimize_hard_interval_by_qp(self, X, y):
        """
        Algorithm1: maximum of hard interval for linear separable
        optimized by QP

        :param X: The feature matrix
        :param y: The lable matrix
        :return: [bias, weights]
        """
        row, col = X.shape

        # initial matrix K's values
        K = np.zeros((row, col + 1))
        for row_index in range(row):
            for col_index in range(col):
                K[row_index, col_index] = -1 * y[row_index, 0] * X[row_index, col_index]
            K[row_index, col] = -1 * y[row_index, 0]

        # initial matrix L's values
        L = np.eye(col + 1) * 0.5
        L[row-1, col] = 0.0

        # initial QP standard matrix
        P = cvxopt.matrix(L)
        q = cvxopt.matrix(np.zeros(row) * 1.0)
        G = cvxopt.matrix(K)
        h = cvxopt.matrix(np.ones(row) * -1.0)

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h)

        print('hello: {}'.format(solution['x']))

        # Intercept
        self.bias = solution['x'][-1]

        # Weight
        self.weights = solution['x'][0:-1]

    def __optimize_soft_interval_by_qp(self, X, y, kernel, C=1.0):
        """
        Algorithm2: maximum of soft interval for linear and non-linear (use kernel conversion) support vector machine 
        optimized by QP

        :param X: The feature matrix
        :param y: The lable matrix
        :param kernel: kernel convertion, one of [linear_kernel, polynomial_kernel, gaussian_kernel]
        :param c: penalty parameter, default 1.0
        :return: [sv_index, sv_x, sv_y, alphas, bias, weights]
        """
        row, col = X.shape

        # Gram matrix
        K = np.zeros((row, row))
        for i in range(row):
            for j in range(row):
                K[i,j] = self.__kernel_convert(X[i], X[j], kernel=kernel)

        # Build QP standard matrix
        P = cvxopt.matrix((np.dot(y, y.T) * K), (row, row), 'd')
        q = cvxopt.matrix(np.ones(row) * -1)
        g1 = np.asarray(np.diag(np.ones(row) * -1))
        g2 = np.asarray(np.diag(np.ones(row)))
        G = cvxopt.matrix(np.append(g1, g2, axis=0))
        h = cvxopt.matrix(np.append(np.zeros(row), (np.ones(row) * C), axis =0))
        A = cvxopt.matrix(np.reshape((y.T), (1,row)), (1, row), 'd')
        b = cvxopt.matrix(np.zeros(1))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        # Lagrange multipliers
        alphas = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = alphas > 1e-5
        self.sv_index = np.arange(len(alphas))[sv]
        self.alphas = alphas[sv]
        self.sv_x = X[sv]
        self.sv_y = y[sv]

        # Intercept
        self.bias = 0
        for i in range(len(self.alphas)):
            self.bias += self.sv_y[i]
            self.bias -= np.sum(self.alphas * self.sv_y * K[self.sv_index[i], sv])
        self.bias /= len(self.alphas)

        # Weight
        if kernel == 'linear_kernel':
            self.weights = np.zeros(col)
            for i in range(len(self.alphas)):
                self.weights += self.alphas[i] * self.sv_y[i] * self.sv_x[i]
        else:
            self.weights = None

    def __optimize_soft_interval_by_smo(self, X, y):
        """
        Algorithm3: maximum of soft interval for linear and non-linear (use kernel conversion) support vector machine 
        optimized by SMO

        :param X: The feature matrix
        :param y: The lable matrix
        """

    def __kernel_convert(self, x1, x2, kernel='linear_kernel'):
        """
        kernel convertion for non-linear

        :param x1: matrix 1
        :param x2: matrix 2
        :param linear_kernel: kernel convertion, one of [linear_kernel, polynomial_kernel, gaussian_kernel]
        """
        sigma=5.0
        p=3

        if kernel == 'linear_kernel':
            return np.dot(x1, x2)
        elif kernel == 'polynomial_kernel':
            return (1 + np.dot(x1, x2)) ** p
        elif kernel == 'gaussian_kernel':
            return np.exp(-np.linalg.norm(x1-x2)**2 / (2 * (sigma ** 2)))
        else:
            return None



if __name__ == '__main__':
    X = pd.DataFrame([[3, 3], [4, 3], [1, 1]]).values
    y = pd.DataFrame([1, 1, -1]).values

    svm = SVM()
    svm.fit(X, y, optimize='hard', kernel='linear_kernel')

    result = svm.predict(np.array([3, 5]))
    probility = svm.predict_probability(np.array([3, 5]))
    print('result is {}'.format(result))
    print('probility is {}'.format(probility))


    print('weight is {}'.format(svm.weights))
    print('bias is {}'.format(svm.bias))
    print('support vector x is {}'.format(svm.sv_x))
    print('support vector y is {}'.format(svm.sv_y))
    print('support vector alphas is {}'.format(svm.alphas))
    print('support vector index is {}'.format(svm.sv_index))
