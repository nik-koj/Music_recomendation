import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from keras.utils import to_categorical
import json

def load_dataset(verbose=0, mode=None, datasetSize=1.0):
    path = "Train_Sliced_Images" if mode == "Train" else "Test_Sliced_Images"
    filenames = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]

    if not filenames:
        raise ValueError("No files for loading. Please check the data directory.")

    images = []
    labels = []

    if mode == "Train":
        metadata_path = 'Dataset/fma_metadata/tracks.csv'
        tracks = pd.read_csv(metadata_path, header=[0, 1], index_col=0)

    for filename in filenames:
        temp = cv2.imread(filename, cv2.IMREAD_UNCHANGED) #!!
        img = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY) #!!
        if img is None:
            continue
        images.append(img)  # Normalization
        #images.append(img / 255.0)
        if mode == "Train":
            # Изменение способа извлечения track_id и genre
            parts = os.path.basename(filename).replace('.jpg', '').split('_')
            track_id = int(parts[0])
            genre = parts[1]  # Теперь genre находится на втором месте

            if track_id in tracks.index:
                genre_label = tracks.loc[track_id, ('track', 'genre_top')]
                if pd.isna(genre_label):
                    continue
                labels.append(genre_label)

    if verbose:
        print(f"Loaded {len(images)} images and {len(labels)} labels")

    images = np.array(images)
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)

    if labels:
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        labels = to_categorical(labels)
        n_classes = labels.shape[1]
        genre_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
        with open("genre_mapping.json", "w") as f:
            json.dump(genre_mapping, f, indent=4)
    else:
        n_classes = 0

    if mode == "Train":
        train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=1 - datasetSize, random_state=42)
        return train_x, train_y, test_x, test_y, n_classes, label_encoder.classes_
    else:
        return images, labels, n_classes, label_encoder.classes_



