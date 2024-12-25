from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Функция для преобразования возрастного диапазона в среднее значение
def age_range_to_mean(age_range):
    if isinstance(age_range, str):
        if '-' in age_range:
            age_min, age_max = map(int, age_range.split('-'))
            return (age_min + age_max) / 2
        elif age_range == '>60':
            return 65

# Определение модели MLP
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

# Загрузка датасета
infrared_thermography_temperature = fetch_ucirepo(id=925)
X = infrared_thermography_temperature.data.features  # type:ignore
y = infrared_thermography_temperature.data.targets['aveOralF']  # type:ignore

# Предобработка данных
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

# Инициализация модели
input_size = X_train.shape[1]
mlp_model = MLP(input_size)

# Определение функции потерь и оптимизатора
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)  # type:ignore
epochs = 5000

# Обучение модели
for epoch in range(epochs):
    mlp_model.train()
    optimizer.zero_grad()
    outputs = mlp_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Оценка модели
mlp_model.eval()
with torch.no_grad():
    y_pred = mlp_model(X_test_tensor)
    mape = torch.mean(torch.abs((y_test_tensor - y_pred) / y_test_tensor)) * 100
    print(f'MAPE (без предобучения): {mape.item():.2f}%')

# Визуализация результатов
y_test_np = y_test_tensor.numpy()
y_pred_np = y_pred.numpy()

plt.figure(figsize=(10, 6))
plt.plot(y_test_np, label='Real Values', color='b')
plt.plot(y_pred_np, label='Predicted Values', color='r', linestyle='dashed')
plt.xlabel('Sample Index')
plt.ylabel('Oral Temperature')
plt.title('Real vs Predicted Values (MLP without Pretraining)')
plt.legend()
plt.show()
