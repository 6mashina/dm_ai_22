# Импорты
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from keras.src.models import Model, Sequential
from keras.src.layers import Input, Dense
from keras.src.optimizers import Adam

# Загрузка данных
cardiotocography = fetch_ucirepo(id=193)
X = cardiotocography.data.features
y = cardiotocography.data.targets

# Предобработка данных
y = y['NSP']
y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Размерность входных данных
input_dim = X_train.shape[1]

# 1. Первый автоэнкодер (входной слой -> скрытый слой 1)
input_layer_1 = Input(shape=(input_dim,))
encoder_1 = Dense(128, activation='relu')(input_layer_1)
decoder_1 = Dense(input_dim, activation='sigmoid')(encoder_1)

autoencoder_1 = Model(inputs=input_layer_1, outputs=decoder_1)
autoencoder_1.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

history_ae1 = autoencoder_1.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Извлекаем кодировщик первого автоэнкодера
encoder_1_model = Model(inputs=input_layer_1, outputs=encoder_1)

# 2. Второй автоэнкодер (скрытый слой 1 -> скрытый слой 2)
input_layer_2 = Input(shape=(128,))
encoder_2 = Dense(64, activation='relu')(input_layer_2)
decoder_2 = Dense(128, activation='sigmoid')(encoder_2)

autoencoder_2 = Model(inputs=input_layer_2, outputs=decoder_2)
autoencoder_2.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Используем представления первого слоя для обучения второго автоэнкодера
encoded_1_train = encoder_1_model.predict(X_train)
history_ae2 = autoencoder_2.fit(
    encoded_1_train, encoded_1_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Извлекаем кодировщик второго автоэнкодера
encoder_2_model = Model(inputs=input_layer_2, outputs=encoder_2)

# 3. Третий автоэнкодер (скрытый слой 2 -> скрытый слой 3)
input_layer_3 = Input(shape=(64,))
encoder_3 = Dense(32, activation='relu')(input_layer_3)
decoder_3 = Dense(64, activation='sigmoid')(encoder_3)

autoencoder_3 = Model(inputs=input_layer_3, outputs=decoder_3)
autoencoder_3.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Используем представления второго слоя для обучения третьего автоэнкодера
encoded_2_train = encoder_2_model.predict(encoded_1_train)
history_ae3 = autoencoder_3.fit(
    encoded_2_train, encoded_2_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Извлекаем кодировщик третьего автоэнкодера
encoder_3_model = Model(inputs=input_layer_3, outputs=encoder_3)

# 4. Собираем полную модель классификации
final_input = Input(shape=(input_dim,))
encoded_1 = encoder_1_model(final_input)  # Первый слой (предобученный)
encoded_2 = encoder_2_model(encoded_1)   # Второй слой (предобученный)
encoded_3 = encoder_3_model(encoded_2)   # Третий слой (предобученный)

# Выходной слой для классификации
output_layer = Dense(len(np.unique(y)), activation='softmax')(encoded_3)

final_model = Model(inputs=final_input, outputs=output_layer)
final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение финальной модели
history_clf = final_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Оценка модели
y_pred_probs = final_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Визуализация результатов
plt.figure(figsize=(18, 8))

# График функции потерь первого автоэнкодера
plt.subplot(2, 2, 1)
plt.plot(history_ae1.history['loss'], label='AE1 Train Loss')
plt.plot(history_ae1.history['val_loss'], label='AE1 Validation Loss')
plt.title('Autoencoder 1 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# График функции потерь второго автоэнкодера
plt.subplot(2, 2, 2)
plt.plot(history_ae2.history['loss'], label='AE2 Train Loss')
plt.plot(history_ae2.history['val_loss'], label='AE2 Validation Loss')
plt.title('Autoencoder 2 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# График функции потерь третьего автоэнкодера
plt.subplot(2, 2, 3)
plt.plot(history_ae3.history['loss'], label='AE3 Train Loss')
plt.plot(history_ae3.history['val_loss'], label='AE3 Validation Loss')
plt.title('Autoencoder 3 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# График точности классификатора
plt.subplot(2, 2, 4)
plt.plot(history_clf.history['accuracy'], label='Classifier Train Accuracy')
plt.plot(history_clf.history['val_accuracy'], label='Classifier Validation Accuracy')
plt.title('Classifier Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
