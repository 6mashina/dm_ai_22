import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv"
data = pd.read_csv(data_url)

X = data.drop(columns=['RMSD']).values  # Признаки
y = data['RMSD'].values  # Целевая переменная


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def to_tensor(data, target):
    return torch.tensor(data, dtype=torch.float32), torch.tensor(target, dtype=torch.float32).view(-1, 1)

X_train_tensor, y_train_tensor = to_tensor(X_train, y_train)
X_test_tensor, y_test_tensor = to_tensor(X_test, y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(X_train.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.1)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v):
        h_prob = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        return h_prob, torch.bernoulli(h_prob)

    def sample_v(self, h):
        v_prob = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        return v_prob, torch.bernoulli(v_prob)

    def forward(self, v):
        _, h = self.sample_h(v)
        return h

    def contrastive_divergence(self, v, k=1):
        v0 = v
        for _ in range(k):
            h_prob, h = self.sample_h(v)
            v_prob, v = self.sample_v(h)
        return v0, v_prob

    def train_rbm(self, train_loader, epochs=10, lr=0.01):
        optimizer = optim.SGD(self.parameters(), lr=lr)
        for epoch in tqdm(range(epochs), desc="Training RBM"):
            epoch_loss = 0
            for batch, _ in train_loader:
                v0, vk = self.contrastive_divergence(batch)
                loss = torch.mean((v0 - vk) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()


def train_model(model, dataloader, criterion, optimizer, epochs=50):
    model.train()
    train_losses = []
    for epoch in tqdm(range(epochs), desc="Training Model"):
        epoch_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(dataloader))
    return train_losses

def evaluate_model(model, dataloader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    return np.sqrt(mean_squared_error(actuals, predictions))


base_model = BaseModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(base_model.parameters(), lr=0.001)

base_train_losses = train_model(base_model, train_loader, criterion, optimizer, epochs=50)
base_rmse = evaluate_model(base_model, test_loader)
print(f"Base Model RMSE: {base_rmse}")


autoencoder = Autoencoder(input_dim=X_train.shape[1])
ae_criterion = nn.MSELoss()
ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)


for epoch in tqdm(range(50), desc="Training Autoencoder"):
    autoencoder.train()
    for inputs, _ in train_loader:
        ae_optimizer.zero_grad()
        _, decoded = autoencoder(inputs)
        loss = ae_criterion(decoded, inputs)
        loss.backward()
        ae_optimizer.step()

pretrained_model = BaseModel()
pretrained_model.model[0].weight.data = autoencoder.encoder[0].weight.data
pretrained_model.model[0].bias.data = autoencoder.encoder[0].bias.data

pretrained_optimizer = optim.Adam(pretrained_model.parameters(), lr=0.001)
pretrained_train_losses = train_model(pretrained_model, train_loader, criterion, pretrained_optimizer, epochs=50)
pretrained_rmse = evaluate_model(pretrained_model, test_loader)
print(f"Pretrained Model RMSE (Autoencoder): {pretrained_rmse}")

rbm = RBM(n_visible=X_train.shape[1], n_hidden=128)
rbm.train_rbm(train_loader, epochs=10, lr=0.01)

rbm_model = BaseModel()
rbm_model.model[0].weight.data = rbm.W.data
rbm_model.model[0].bias.data = rbm.h_bias.data


rbm_optimizer = optim.Adam(rbm_model.parameters(), lr=0.001)
rbm_train_losses = train_model(rbm_model, train_loader, criterion, rbm_optimizer, epochs=50)
rbm_rmse = evaluate_model(rbm_model, test_loader)
print(f"Pretrained Model RMSE (RBM): {rbm_rmse}")

print(f"Improvement (Autoencoder): {base_rmse - pretrained_rmse}")
print(f"Improvement (RBM): {base_rmse - rbm_rmse}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, 51), base_train_losses, label='Base Model')
plt.plot(range(1, 51), pretrained_train_losses, label='Pretrained Model (Autoencoder)')
plt.plot(range(1, 51), rbm_train_losses, label='Pretrained Model (RBM)')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss by Epoch')
plt.legend()
plt.show()
