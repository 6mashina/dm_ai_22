import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import torch.nn.functional as F


# Фиксируем seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# Загружаем данные
infrared_thermography_temperature = fetch_ucirepo(id=925)
X = infrared_thermography_temperature.data.features
y = infrared_thermography_temperature.data.targets

y = y['aveOralF']  # Или 'aveOralM'

for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

X = X.fillna(X.mean())
y = y.fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"X_train_tensor shape: {X_train_tensor.shape}")


class RBM(nn.Module):
    def __init__(self, visible_dim, hidden_dim):
        super(RBM, self).__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.W = nn.Parameter(torch.randn(visible_dim, hidden_dim) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(visible_dim))
        self.h_bias = nn.Parameter(torch.zeros(hidden_dim))

    def sample_h(self, v):
        # ReLU активация для скрытых нейронов
        h_activation = F.relu(torch.matmul(v, self.W) + self.h_bias)
        h_sample = h_activation + torch.randn_like(h_activation) * 0.01
        return h_activation, h_sample

    def sample_v(self, h):
        # ReLU активация для видимых нейронов
        v_activation = F.relu(torch.matmul(h, self.W.t()) + self.v_bias)
        v_sample = v_activation + torch.randn_like(v_activation) * 0.01
        return v_activation, v_sample

    def forward(self, v):
        h_activation, h_sample = self.sample_h(v)
        v_activation, v_sample = self.sample_v(h_sample)
        return v_sample, v_activation, h_activation

    def contrastive_divergence(self, v, k=1):
        v0 = v
        for _ in range(k):
            _, h_sample = self.sample_h(v0)
            v_activation, v0 = self.sample_v(h_sample)
        return v0


def pretrain_rbm_stack(train_data, rbm_layers, epochs, lr):
    input_data = train_data
    trained_rbms = []

    for idx, rbm in enumerate(rbm_layers):
        print(f"Training RBM Layer {idx + 1}")
        optimizer = optim.SGD(rbm.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss = 0
            for batch in DataLoader(TensorDataset(input_data), batch_size=64, shuffle=True):
                v_batch = batch[0]
                assert v_batch.shape[
                           1] == rbm.visible_dim, f"Input dim mismatch: {v_batch.shape[1]} != {rbm.visible_dim}"

                optimizer.zero_grad()
                v0 = v_batch
                vk = rbm.contrastive_divergence(v0)
                loss = torch.mean((v0 - vk) ** 2)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"RBM Layer {idx + 1}, Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

        trained_rbms.append(rbm)

        with torch.no_grad():
            _, h_sample = rbm.sample_h(input_data)
            input_data = h_sample
            print(f"Output shape after RBM Layer {idx + 1}: {input_data.shape}")

    return trained_rbms


visible_dim = X_train.shape[1]
hidden_dims = [128, 64, 32]

rbm_layers = [RBM(visible_dim, hidden_dims[0])]
for i in range(len(hidden_dims) - 1):
    rbm_layers.append(RBM(hidden_dims[i], hidden_dims[i + 1]))


class CNNRegressor(nn.Module):
    def __init__(self, encoder):
        super(CNNRegressor, self).__init__()
        self.encoder = encoder
        self.regressor = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        for rbm in self.encoder:
            _, x = rbm.sample_h(x)
        return self.regressor(x)


visible_dim = X_train.shape[1]
hidden_dims = [128, 64, 32]

rbm_layers = [RBM(visible_dim, hidden_dims[0])]
for i in range(len(hidden_dims) - 1):
    rbm_layers.append(RBM(hidden_dims[i], hidden_dims[i + 1]))

trained_rbms = pretrain_rbm_stack(X_train_tensor, rbm_layers, epochs=20, lr=0.01)

cnn_model = CNNRegressor(trained_rbms)  # Используем обученный стек RBM
criterion = nn.MSELoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.0001)

epochs = 1000
train_losses = []

for epoch in range(epochs):
    cnn_model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = cnn_model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))
    if (epoch + 1) % 100 == 0:
        print(f"Regression Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

plt.plot(range(1, epochs + 1), train_losses, label="Regression Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Regression Training Loss vs Epochs")
plt.show()

cnn_model.eval()
y_pred = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        preds = cnn_model(X_batch)
        y_pred.extend(preds.numpy())

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Истинные значения", color="blue")
plt.plot(y_pred, label="Прогнозируемые значения", color="orange")
plt.xlabel("Образцы")
plt.ylabel("Значение")
plt.title("Прогнозируемые vs Истинные значения")
plt.legend()
plt.show()

mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE: {mape: .4f}")
