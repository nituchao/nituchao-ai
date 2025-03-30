import numpy as np

# 定义目标函数
def objective_function(x):
    return 3*x[0] + 2*x[1]

# 定义约束条件
def constraint(x):
    return x[0] + x[1] - 2

# 拉格朗日松弛函数
def lagrangian_relaxation(x, lambda_):
    return objective_function(x) + lambda_ * constraint(x)

# 初始解
x0 = np.array([0.5, 1.5])
lambda_ = 1.0  # 拉格朗日乘子

# 进行迭代
for _ in range(10):
    # 计算松弛后的目标值
    relaxed_value = lagrangian_relaxation(x0, lambda_)

    # 根据松弛结果更新解
    # 这里简单地随机更新
    x0 = np.random.rand(2)

# 输出最终结果
print("最终解:", x0)
print("最终目标值:", objective_function(x0))
print("最终lambda值:", lambda_)