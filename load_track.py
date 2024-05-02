import os
import pandas as pd
import numpy as np
import cv2
from keras.models import load_model
from keras.models import Model
from single_song_processor import create_and_slice_spectrogram
import sqlite3
import numpy as np
import json


def load_genre_mapping(genre_file='genres.json'):
    with open(genre_file, 'r') as file:
        genre_mapping = json.load(file)
    return genre_mapping


def load_models(model_path):
    full_model = load_model(model_path)
    feature_model = Model(inputs=full_model.inputs, outputs=full_model.layers[-2].output)  # Предпоследний слой для признаков
    return full_model, feature_model

def parse_filename(file_path):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    parts = base_name.split('-')
    title = parts[0].strip()
    artist = parts[1].strip() if len(parts) > 1 else 'Unknown Artist'
    return title, artist

def insert_track_data(title, artist, genre_top, features, db_path="music_features.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO features (title, artist, genre_top, features)
        VALUES (?, ?, ?, ?)
    ''', (title, artist, genre_top, ','.join(map(str, features))))
    conn.commit()
    conn.close()
def predict_genre_and_features(slices_dir, full_model, feature_model, genre_mapping):
    filenames = [os.path.join(slices_dir, f) for f in os.listdir(slices_dir) if f.endswith(".jpg")]
    genre_votes = {}
    all_features = []

    for filename in filenames:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(np.expand_dims(img, axis=-1), axis=0) / 255.0

        genre_probabilities = full_model.predict(img)
        predicted_genre_index = np.argmax(genre_probabilities, axis=-1)[0]
        predicted_genre = genre_mapping[str(predicted_genre_index)]
        genre_votes[predicted_genre] = genre_votes.get(predicted_genre, 0) + 1

        features = feature_model.predict(img)[0]
        all_features.append(features)

    predominant_genre = max(genre_votes, key=genre_votes.get)
    avg_features = np.mean(all_features, axis=0)
    return predominant_genre, avg_features.tolist()


def process_track(file_path, output_dir, db_path, full_model, feature_model):
    title, artist = parse_filename(file_path)
    genre_mapping = load_genre_mapping()  # Загрузка маппинга жанров
    create_and_slice_spectrogram(file_path, output_dir)
    genre_top, features = predict_genre_and_features(output_dir, full_model, feature_model, genre_mapping)
    insert_track_data(title, artist, genre_top, features, db_path)


# Основной блок исполнения
if __name__ == "__main__":
    model_path = "best_model.keras"
    full_model, feature_model = load_models(model_path)
    file_path = 'Dataset/fma_small/108464.mp3'
    output_dir = 'Song_Spectrograms'
    process_track(file_path, output_dir, "music_features.db", full_model, feature_model)