import random

class PIDController:
    def __init__(self, kp, ki, kd, setpoint, dt):
        self.kp = kp  # 比例系数
        self.ki = ki  # 积分系数
        self.kd = kd  # 微分系数
        self.setpoint = setpoint  # 目标值
        self.dt = dt  # 时间间隔
        self.error_sum = 0  # 误差积分
        self.last_error = 0  # 上一次误差

    def calculate_output(self, current_value):
        # 计算误差
        error = self.setpoint - current_value

        # 计算积分项
        self.error_sum += error * self.dt

        # 计算微分项
        derivative = (error - self.last_error) / self.dt

        # 计算输出值
        output = self.kp * error + self.ki * self.error_sum + self.kd * derivative

        # 更新上一次误差
        self.last_error = error

        return output

# 模拟广告成本流量控制
def simulate_ad_cost_control():
    # 初始化PID控制器
    pid = PIDController(kp=0.5, ki=0.1, kd=0.05, setpoint=100, dt=1)

    # 模拟广告成本
    ad_cost = 50

    # 模拟广告流量
    ad_traffic = 100

    # 模拟广告效果
    conversion_rate = 0.05

    # 模拟广告投放过程
    for i in range(10):
        # 计算广告成本偏差
        error = pid.setpoint - ad_cost

        # 使用PID控制器计算输出值
        output = pid.calculate_output(ad_cost)

        # 调整广告预算
        ad_budget = output

        # 模拟广告投放结果
        ad_cost = random.randint(40, 60)  # 模拟广告成本波动
        ad_traffic = ad_budget * conversion_rate  # 模拟广告流量变化

        print(f"Iteration {i+1}:")
        print(f"  Ad Cost: {ad_cost}")
        print(f"  Ad Traffic: {ad_traffic}")
        print(f"  Ad Budget: {ad_budget}")

if __name__ == "__main__":
    simulate_ad_cost_control()