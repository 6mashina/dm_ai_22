# Импорт необходимых библиотек
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import BernoulliRBM
from keras.src.models import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

# Загрузка данных
from ucimlrepo import fetch_ucirepo

cardiotocography = fetch_ucirepo(id=193)
X = cardiotocography.data.features
y = cardiotocography.data.targets['NSP']
y = (y == 2).astype(int)

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Предобучение каждого слоя с помощью RBM
layer_1 = BernoulliRBM(n_components=128, learning_rate=0.01, n_iter=10, random_state=42)
layer_2 = BernoulliRBM(n_components=64, learning_rate=0.01, n_iter=10, random_state=42)
layer_3 = BernoulliRBM(n_components=32, learning_rate=0.01, n_iter=10, random_state=42)

# Обучение RBM-слоев
X_layer_1 = layer_1.fit_transform(X_train)  # Первый слой
X_layer_2 = layer_2.fit_transform(X_layer_1)  # Второй слой
X_layer_3 = layer_3.fit_transform(X_layer_2)  # Третий слой

# Инициализация MLP с использованием предобученных весов
model = Sequential()

# Первый слой (инициализируем предобученными весами RBM)
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.layers[-1].set_weights([layer_1.components_.T, np.zeros(128)])

# Второй слой (инициализируем предобученными весами RBM)
model.add(Dense(64, activation='relu'))
model.layers[-1].set_weights([layer_2.components_.T, np.zeros(64)])

# Третий слой (инициализируем предобученными весами RBM)
model.add(Dense(32, activation='relu'))
model.layers[-1].set_weights([layer_3.components_.T, np.zeros(32)])

# Выходной слой
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Оценка модели
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Матрица конфузии
print("\nМатрица конфузии:")
print(confusion_matrix(y_test, y_pred))

# Отчет о классификации
print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))

# Визуализация потерь и точности
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Потери
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Ошибка')
plt.xlabel('Эпоха')
plt.ylabel('Ошибка')
plt.legend()

# Точность
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Точность')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()

plt.tight_layout()
plt.show()
