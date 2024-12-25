import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 1. Загрузка и подготовка данных
data = pd.read_csv('forest+fires/forestfires.csv')

# Проверка типов данных и уникальных значений для анализа
print(data.dtypes)
print(data.isna().sum())
print(data.head())

# Вывод уникальных значений для столбцов month и day
print("Уникальные значения в столбце month:", data['month'].unique())
print("Уникальные значения в столбце day:", data['day'].unique())

# Удаление ненужных столбцов и преобразование в числовой формат
data.drop(['month', 'day'], axis=1, inplace=True)

# Преобразование всех оставшихся столбцов в числовой формат
data = data.apply(pd.to_numeric, errors='coerce')

# Проверка количества строк до и после удаления NaN
print(f"Количество строк до удаления NaN: {len(data)}")
data.dropna(inplace=True)
print(f"Количество строк после удаления NaN: {len(data)}")

# Проверка оставшихся строк
if len(data) == 0:
    raise ValueError("Все строки были удалены, нет данных для обучения.")

# Убедитесь, что целевая переменная 'area' присутствует
if 'area' not in data.columns:
    raise ValueError("Целевая переменная 'area' не найдена в данных.")

# Поделите данные на признаки и целевую переменную
X = data.drop('area', axis=1).values
y = data['area'].values

# Проверка на наличие строк в X и y
if X.shape[0] == 0 or y.shape[0] == 0:
    raise ValueError("Нет данных для обучения.")

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Обучение модели без предобучения
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Выходной слой для регрессии
])

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_percentage_error'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=32)

# Оценка модели
loss, mape = model.evaluate(X_test, y_test)
print(f"Test MAPE (без предобучения): {mape}")

# Визуализация истории обучения
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss (без предобучения)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

# 3. Обучение с предобучением с использованием автоэнкодера
# Создание автоэнкодера
input_layer = layers.Input(shape=(X_train.shape[1],))
encoded = layers.Dense(64, activation='relu')(input_layer)
encoded = layers.Dense(32, activation='relu')(encoded)
decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(X_train.shape[1], activation='linear')(decoded)

autoencoder = keras.Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Обучение автоэнкодера
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2)

# Извлечение закодированных признаков
encoder = keras.Model(input_layer, encoded)
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# Обучение модели с закодированными признаками
model_encoded = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_encoded.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Выходной слой для регрессии
])

# Компиляция модели
model_encoded.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_percentage_error'])

# Обучение модели
history_encoded = model_encoded.fit(X_train_encoded, y_train, epochs=100, validation_split=0.2, batch_size=32)

# Оценка модели
loss_encoded, mape_encoded = model_encoded.evaluate(X_test_encoded, y_test)
print(f"Test MAPE (с предобучением): {mape_encoded}")

# Визуализация истории обучения автоэнкодера
plt.subplot(1, 2, 2)
plt.plot(history_encoded.history['loss'], label='loss')
plt.plot(history_encoded.history['val_loss'], label='val_loss')
plt.title('Autoencoder Model Loss (с предобучением)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()
