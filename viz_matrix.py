import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from load_data import load_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import random
import json

# Загрузка и подготовка данных
train_x, train_y, test_x, test_y, n_classes, genre = load_dataset(verbose=1, mode="Train", datasetSize=0.8)

# Нормализация данных
test_x = test_x / 255.0

# Загрузка модели
model = load_model("best_model.keras")

# Предсказания на тестовых данных
predictions = model.predict(test_x, verbose=0)

# Загрузка маппинга жанров
with open('Mapping.json', 'r', encoding='utf-8') as f:
    genre_mapping = json.load(f)

# Обратный маппинг жанров
genre_inverse_mapping = {v: k for k, v in genre_mapping.items()}

# Визуализация случайных предсказаний
sample_indices = random.sample(range(test_x.shape[0]), 15)
sample_images = [test_x[i].reshape(128, 128) for i in sample_indices]
sample_labels = [np.argmax(test_y[i]) for i in sample_indices]
predicted = [np.argmax(predictions[i]) for i in sample_indices]

fig, axes = plt.subplots(5, 3, figsize=(15, 10))
for i, ax in enumerate(axes.flat):
    truth = sample_labels[i]
    truth_genre = genre_inverse_mapping[truth]
    prediction = predicted[i]
    prediction_genre = genre_inverse_mapping[prediction]
    ax.axis('off')
    ax.imshow(sample_images[i], cmap='gray')
    color = 'green' if truth == prediction else 'red'
    ax.set_title(f"Истина: {truth_genre}\nПредсказание: {prediction_genre}", fontsize=10, color=color, loc='left')
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.savefig("Graphs/Training_Prediction.jpg")
plt.show()
plt.close()

# Создание и визуализация матрицы ковариаций
y_pred = np.argmax(predictions, axis=1)
y_test = np.argmax(test_y, axis=1)
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Отображение матрицы ковариаций с численными показателями
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_normalized, display_labels=[genre_inverse_mapping[i] for i in range(n_classes)])
fig, ax = plt.subplots(figsize=(10, 10))  # Увеличиваем размер фигуры для лучшего отображения
disp.plot(cmap='viridis', xticks_rotation=45, ax=ax)
plt.title('Матрица ковариаций с вероятностями')
plt.tight_layout()  # Убедитесь, что компоненты не перекрываются
plt.savefig("Graphs/Confusion_Matrix.png")
plt.show()

print("Матрица ковариаций создана")