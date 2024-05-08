import os
import pandas as pd
import numpy as np
import cv2
from keras.models import load_model
from keras.models import Model
import sqlite3
import numpy as np


def load_feature_extraction_model(model_path):
    model = load_model(model_path)
    feature_model = Model(inputs=model.inputs, outputs=model.layers[-2].output)  # Используем выход с предпоследнего слоя
    return feature_model

def create_database(db_path="music_features.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY,
            title TEXT,
            artist TEXT,
            genre_top TEXT,
            features TEXT,
            file_path TEXT
        )
    ''')
    conn.commit()
    conn.close()


def insert_data(data, db_path="music_features.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for index, row in data.iterrows():
        cursor.execute('''
            INSERT INTO features (title, artist, genre_top, features, file_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (row['title'], row['artist'], row['genre_top'], ','.join(map(str, row['features'])), row['file_path']))
    conn.commit()
    conn.close()


def extract_and_aggregate_features(verbose=0, mode=None, feature_model=None, db_path="music_features.db"):
    path = "Train_Sliced_Images" if mode == "Train" else "Test_Sliced_Images"
    metadata_path = 'Dataset/fma_metadata/tracks.csv'
    tracks = pd.read_csv(metadata_path, header=[0, 1], index_col=0)

    filenames = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
    features_dict = {}

    for filename in filenames:
        try:
            parts = os.path.basename(filename).replace('.jpg', '').split('_')
            track_id = int(parts[0])
            if track_id not in features_dict:
                track_info = tracks.loc[track_id]
                file_path = os.path.join('Dataset', 'fma_small', f'{str(track_id).zfill(6)}.mp3')  # Правильный путь
                features_dict[track_id] = {
                    'title': track_info[('track', 'title')],
                    'artist': track_info[('artist', 'name')],
                    'genre_top': track_info[('track', 'genre_top')],
                    'features': [],
                    'file_path': file_path
                }

            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0) / 255.0
            features = feature_model.predict(img)[0]
            features_dict[track_id]['features'].append(features)

            if verbose:
                print(f"Processed {filename} successfully")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    # Усреднение признаков и подготовка данных для сохранения
    aggregated_data = []
    for track_id, info in features_dict.items():
        avg_features = np.mean(info['features'], axis=0)
        aggregated_data.append({
            'title': info['title'],
            'artist': info['artist'],
            'genre_top': info['genre_top'],
            'features': avg_features.tolist(),
            'file_path': info['file_path']  # Убедитесь, что используется правильный путь
        })

    return pd.DataFrame(aggregated_data)


if __name__ == "__main__":
    create_database()
    model_path = "best_model.keras"  # Путь к сохраненной модели
    feature_model = load_feature_extraction_model(model_path)
    feature_data = extract_and_aggregate_features(verbose=1, mode='Train', feature_model=feature_model)
    insert_data(feature_data)