import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from keras.initializers import he_normal
from load_data import load_dataset
import pandas as pd
import json


def save_genre_mapping(label_encoder, filename='genre_mapping.json'):
    genre_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    with open(filename, 'w') as f:
        json.dump(genre_mapping, f)

# Загрузка и подготовка данных
train_x, train_y, test_x, test_y, n_classes, genre = load_dataset(verbose=1, mode="Train", datasetSize=0.75)
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1) / 255.0
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1) / 255.0

# Построение модели
model = Sequential([
    Input(shape=(128, 128, 1)),
    Conv2D(64, (7, 7), activation='relu',  kernel_initializer=he_normal(seed=1)),
    BatchNormalization(),
    AveragePooling2D((2, 2), strides=2),
    Conv2D(128, (7, 7), strides=2, activation='relu', kernel_initializer=he_normal(seed=1)),
    BatchNormalization(),
    AveragePooling2D((2, 2), strides=2),
    Conv2D(256, (3, 3), activation='relu', kernel_initializer=he_normal(seed=1)),
    BatchNormalization(),
    AveragePooling2D((2, 2), strides=2),
    Conv2D(512, (3, 3), activation='relu', kernel_initializer=he_normal(seed=1)),
    BatchNormalization(),
    AveragePooling2D((2, 2), strides=2),
    Flatten(),
    BatchNormalization(),
    Dropout(0.6),
    Dense(1024, activation='relu', kernel_initializer=he_normal(seed=1)),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_initializer=he_normal(seed=1)),
    Dropout(0.25),
    Dense(64, activation='relu', kernel_initializer=he_normal(seed=1)),
    Dense(32, activation='relu', kernel_initializer=he_normal(seed=1)),
    Dense(n_classes, activation='softmax', kernel_initializer=he_normal(seed=1))
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)

# Обучение
history = model.fit(
    train_x,
    train_y,
    epochs=50,  # Увеличил количество эпох для возможности ранней остановки
    batch_size=32,
    verbose=1,
    validation_split=0.1,
    callbacks=[checkpoint, early_stopping]
)

train_history = pd.DataFrame(history.history)
train_history.to_csv("training_history.csv")
try:
    save_genre_mapping()
except Exception as e:
    print(f"Failed to process: {e}")
# Оценка
score = model.evaluate(test_x, test_y)
print("Test accuracy:", score[1])


# Сохранение и визуализация архитектуры
model.save("Saved_Model/Model.keras")
plot_model(model, to_file="Model_Architecture.png")