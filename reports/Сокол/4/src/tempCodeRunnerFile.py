import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Функция обработки возрастного диапазона
def age_range_to_mean(age_range):
    if isinstance(age_range, str):
        if '-' in age_range:
            age_min, age_max = map(int, age_range.split('-'))
            return (age_min + age_max) / 2
        elif age_range == '>60':
            return 65

# Определение RBM
class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units, k=1):
        super(RBM, self).__init__()
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.k = k

        self.W = torch.randn(hidden_units, visible_units) * 0.1
        self.h_bias = torch.zeros(hidden_units)
        self.v_bias = torch.zeros(visible_units)

    def sample_h(self, v):
        h_prob = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        return h_prob, torch.bernoulli(h_prob)

    def sample_v(self, h):
        v_prob = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        return v_prob, torch.bernoulli(v_prob)

    def forward(self, v):
        h_prob, h_sample = self.sample_h(v)
        for _ in range(self.k):
            v_prob, v_sample = self.sample_v(h_sample)
            h_prob, h_sample = self.sample_h(v_sample)
        return v_prob

    def train_rbm(self, data, epochs=10, lr=0.01):
        for epoch in range(epochs):
            v0 = data
            h_prob, h_sample = self.sample_h(v0)
            v1_prob, v1 = self.sample_v(h_sample)
            h1_prob, _ = self.sample_h(v1)

            positive_grad = torch.matmul(h_prob.t(), v0)
            negative_grad = torch.matmul(h1_prob.t(), v1)

            self.W += lr * (positive_grad - negative_grad) / data.size(0)
            self.v_bias += lr * torch.sum(v0 - v1, dim=0) / data.size(0)
            self.h_bias += lr * torch.sum(h_prob - h1_prob, dim=0) / data.size(0)

            loss = torch.mean(torch.sum((v0 - v1_prob) ** 2, dim=1))
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Определение MLP
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Предобучение RBM для слоёв MLP
def pretrain_rbm_on_mlp(X_train_tensor, layer_sizes):
    rbms = []
    prev_size = X_train_tensor.shape[1]

    for size in layer_sizes:
        rbm = RBM(visible_units=prev_size, hidden_units=size)
        rbm.train_rbm(X_train_tensor, epochs=50, lr=0.01)
        rbms.append(rbm)

        with torch.no_grad():
            _, hidden = rbm.sample_h(X_train_tensor)
            X_train_tensor = hidden
        prev_size = size

    return rbms

# Загрузка данных
infrared_thermography_temperature = fetch_ucirepo(id=925)
X = infrared_thermography_temperature.data.features  # type:ignore
y = infrared_thermography_temperature.data.targets['aveOralF']  # type:ignore

X.dropna(inplace=True)
y = y[X.index]
X['Age'] = X['Age'].apply(age_range_to_mean)

categorical_features = ['Gender', 'Ethnicity']
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

visible_units = X_train_tensor.shape[1]
layer_sizes = [64, 32, 16]
rbms = pretrain_rbm_on_mlp(X_train_tensor, layer_sizes)

mlp_model = MLP(input_size=visible_units)

mlp_model.fc1.weight.data = rbms[0].W
mlp_model.fc1.bias.data = rbms[0].h_bias

mlp_model.fc2.weight.data = rbms[1].W
mlp_model.fc2.bias.data = rbms[1].h_bias

mlp_model.fc3.weight.data = rbms[2].W
mlp_model.fc3.bias.data = rbms[2].h_bias

criterion = nn.MSELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

# Обучение MLP
epochs = 2000
for epoch in range(epochs):
    mlp_model.train()
    optimizer.zero_grad()
    outputs = mlp_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Вычисление MAPE
mlp_model.eval()
with torch.no_grad():
    y_pred = mlp_model(X_test_tensor)
    mape = torch.mean(torch.abs((y_test_tensor - y_pred) / y_test_tensor)) * 100
    print(f'MAPE (с предобучением): {mape.item():.2f}%')

# График реальных и предсказанных значений
y_test_np = y_test_tensor.numpy()
y_pred_np = y_pred.numpy()

plt.figure(figsize=(10, 6))
plt.plot(y_test_np, label='Real Values', color='b')
plt.plot(y_pred_np, label='Predicted Values', color='r', linestyle='dashed')
plt.xlabel('Sample Index')
plt.ylabel('Oral Temperature')
plt.title('Real vs Predicted Values (MLP with RBM Pretraining)')
plt.legend()
plt.show()
