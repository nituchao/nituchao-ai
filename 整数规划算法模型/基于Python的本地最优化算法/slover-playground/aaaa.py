import numpy as np
from scipy.optimize import minimize

# Define the integer programming problem
n = 4  # number of variables
c = np.array([3, 2, 4, 1])  # objective function coefficients
A = np.array([[2, 1, 1, 1], [1, 2, 1, 1]])  # constraint matrix
b = np.array([6, 5])  # constraint right-hand side

# Lagrangian relaxation function
def lagrangian_relaxation(x, lam):
    L = np.dot(c, x) + np.dot(lam, b - np.dot(A, x))
    return L

# Lagrangian subproblem solver
def solve_subproblem(lam):
    x = np.maximum(0, np.floor(c - np.dot(A.T, lam)))
    return x

# Dual function
def dual_function(lam):
    x = solve_subproblem(lam)
    return lagrangian_relaxation(x, lam)

# Optimize the Lagrangian dual problem
lam0 = np.zeros(A.shape[0])  # Initialize Lagrange multipliers
res = minimize(dual_function, lam0, method='L-BFGS-B', bounds=[(0, None)] * A.shape[0])
lam_star = res.x

# Compute the optimal integer solution
x_star = solve_subproblem(lam_star)
obj_value = np.dot(c, x_star)

print("Optimal integer solution:", x_star.astype(int))
print("Optimal objective value:", obj_value)
print("Optimal lambda0 value:", lam0)