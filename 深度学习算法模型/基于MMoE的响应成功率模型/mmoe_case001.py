import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 定义 MMoE 模型
class MMoE(nn.Module):
    def __init__(self, num_experts, num_tasks, hidden_size, expert_hidden_size):
        super(MMoE, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.hidden_size = hidden_size
        self.expert_hidden_size = expert_hidden_size

        # 专家网络
        self.experts = nn.ModuleList([nn.Linear(hidden_size, expert_hidden_size) for _ in range(num_experts)])
        # 门控网络
        self.gates = nn.ModuleList([nn.Linear(hidden_size, num_experts) for _ in range(num_tasks)])
        # 任务网络
        self.tasks = nn.ModuleList([nn.Linear(expert_hidden_size, 1) for _ in range(num_tasks)])

    def forward(self, x):
        # 专家网络输出
        expert_outputs = [expert(x) for expert in self.experts]
        # 门控网络输出
        gate_outputs = [gate(x) for gate in self.gates]

        # 计算门控权重
        gate_weights = [torch.softmax(gate_output, dim=1) for gate_output in gate_outputs]

        # 加权求和
        task_outputs = []
        for i in range(self.num_tasks):
            # 获取每个任务的专家输出
            task_expert_outputs = [expert_outputs[j] * gate_weights[i][:, j:j+1] for j in range(self.num_experts)]
            # 将专家输出加权求和
            task_output = torch.sum(torch.stack(task_expert_outputs), dim=0)
            # 任务网络输出
            task_output = self.tasks[i](task_output)
            task_outputs.append(task_output)

        return task_outputs

# 定义数据集
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 样本运行例子
if __name__ == '__main__':
    # 设置参数
    num_experts = 5
    num_tasks = 2
    hidden_size = 10
    expert_hidden_size = 5
    batch_size = 32
    learning_rate = 0.01
    epochs = 10

    # 生成样本数据
    data = torch.randn(100, hidden_size)
    # 创建数据集
    dataset = MyDataset(data)
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = MMoE(num_experts, num_tasks, hidden_size, expert_hidden_size)
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 初始化损失函数
    criterion = nn.MSELoss()

    # 训练模型
    for epoch in range(epochs):
        for batch_idx, data in enumerate(dataloader):
            # 前向传播
            outputs = model(data)
            # 计算损失
            loss = criterion(outputs[0], torch.randn(batch_size, 1)) + criterion(outputs[1], torch.randn(batch_size, 1))
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 更新参数
            optimizer.step()

            # 打印训练进度
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')

    # 测试模型
    with torch.no_grad():
        test_data = torch.randn(10, hidden_size)
        test_outputs = model(test_data)
        print(f'Test Outputs: {test_outputs}')