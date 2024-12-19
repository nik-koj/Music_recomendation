import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Загрузка истории обучения
train_history = pd.read_csv("training_history.csv")

# Графики точности и потерь
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_history['accuracy'], label='Точность на обучающей выборке')
plt.plot(train_history['val_accuracy'], label='Точность на валидационной выборке')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.title('Точность по эпохам')
plt.legend()
# Явно задать метки для оси X
plt.xticks(range(0, len(train_history['accuracy']), 2))

plt.subplot(1, 2, 2)
plt.plot(train_history['loss'], label='Потери на обучающей выборке')
plt.plot(train_history['val_loss'], label='Потери на валидационной выборке')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.title('Потери по эпохам')
plt.legend()
# Явно задать метки для оси X
plt.xticks(range(0, len(train_history['loss']), 2))

plt.show()



#