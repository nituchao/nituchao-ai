import numpy as np
import pandas as pd
import cvxopt

class SVM:

    def __init__(self):
        return

    # 使用二次规划解最大间隔法，对应线性可分支持向量机
    def optimize_max_interval_by_qp(self, X, y):
        row, col = X.shape

        # initial matrix K's values
        K = np.zeros((row, col + 1))
        for row_index in range(row):
            for col_index in range(col):
                print('({}, {}): -1 * y({}) * X({})'.format(row_index, col_index, y[row_index, 0], X[row_index, col_index]))
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

        print('P is {}'.format(P))
        print('q is {}'.format(q))
        print('G is {}'.format(G))
        print('h is {}'.format(h))

        # optimize by cvxopt qp
        solution = cvxopt.solvers.qp(P, q, G, h)

        return solution['x']

    # 使用二次规划解软间隔法，对应线性支持向量机
    def optimize_soft_interval_by_qp(self, X, y, c):
        m = len(X)

        P = cvxopt.matrix((np.dot(y, y.T) * np.dot(X, X.T)), (m, m), 'd')
        q = cvxopt.matrix(np.ones(m) * -1)
        g1 = np.asarray(np.diag(np.ones(m) * -1))
        g2 = np.asarray(np.diag(np.ones(m)))
        G = cvxopt.matrix(np.append(g1, g2, axis=0))
        h = cvxopt.matrix(np.append(np.zeros(m), (np.ones(m) * c), axis =0))
        A = cvxopt.matrix(np.reshape((y.T), (1,m)), (1, m), 'd')
        b = cvxopt.matrix(np.zeros(1))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        # Lagrange multipliers
        a = np.ravel(solution['x'])

        

if __name__ == '__main__':
    X = pd.DataFrame([[3.0, 3.0], [4.0, 3.0], [1.0, 1.0]]).values
    y = pd.DataFrame([1.0, 1.0, -1.0]).values

    svm = SVM()
    svm.optimize_soft_interval_by_qp(X, y, 2)
