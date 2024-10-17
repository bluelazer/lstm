import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 定义超参数
input_size = 4  # 输入特征数：NOx浓度、锅炉温度、风速、氧气浓度
hidden_size = 64  # LSTM的隐藏层大小
num_layers = 2  # LSTM的层数
output_size = 10  # 预测未来10分钟的NOx浓度
seq_length = 60  # 输入前60分钟的数据
learning_rate = 0.001
batch_size = 32
num_epochs = 20
validation_split = 0.2  # 20% 数据作为验证集

# 定义自定义数据集类
class NOxDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.Tensor(self.data[index]), torch.Tensor(self.target[index])

# 构建LSTM模型
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))  # LSTM前向传播
        out = self.fc(out[:, -1, :])  # 使用最后一个时间步的输出
        return out

# 数据预处理 (将你已有的历史数据进行分割)
def create_sequences(data, seq_length, output_length):
    x, y = [], []
    for i in range(len(data) - seq_length - output_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+output_length, 0])  # 只预测NOx浓度
    return np.array(x), np.array(y)

# 加载数据
path = r"G:\lstm\data_p\data_50_0615.csv"
df = pd.read_csv(path)
features = ['ba1_cemscom_rin_1_meas', 'ba1_dcs_110113_04_pnt', 'ba1_cems_01_03_rtd', 'ba1_dcs_110111_02_pnt']
df = df[features]
historical_data = df.values

# 归一化数据
scaler = MinMaxScaler()
historical_data = scaler.fit_transform(historical_data)

# 构建输入序列和目标序列
x_data, y_data = create_sequences(historical_data, seq_length, output_size)

# 分割训练集和验证集
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=validation_split, shuffle=True)

# 构建数据集和数据加载器
train_dataset = NOxDataset(x_train, y_train)
val_dataset = NOxDataset(x_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMPredictor(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()  # 使用均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
model.train()
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 验证集上的表现
    model.eval()  # 进入评估模式
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()

    val_loss /= len(val_loader)
    print(f'Validation Loss: {val_loss:.4f}')
    model.train()  # 重新进入训练模式

# 保存模型
torch.save(model.state_dict(), 'nox_lstm_model.pth')

# 预测函数
def predict(model, input_data):
    model.eval()
    with torch.no_grad():
        input_data = torch.Tensor(input_data).unsqueeze(0).to(device)
        prediction = model(input_data)
    return prediction.cpu().numpy()

# 使用模型进行预测
example_input = historical_data[-seq_length:]  # 最后60分钟的数据
prediction = predict(model, example_input)
print('未来10分钟NOx浓度预测：', prediction)
