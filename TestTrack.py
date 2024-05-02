import os
import numpy as np
import json
from keras.models import load_model
import cv2


def load_model_for_genre_prediction(model_path):
    return load_model(model_path)


def load_genre_mapping(genre_file='genres.json'):
    with open(genre_file, 'r') as file:
        genre_mapping = json.load(file)
    return genre_mapping


def load_and_predict_genre(model, spectrogram_dir, genre_mapping):
    filenames = [os.path.join(spectrogram_dir, f) for f in os.listdir(spectrogram_dir) if f.endswith('.jpg')]
    genre_votes = {}

    for filename in filenames:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(np.expand_dims(img, axis=-1), axis=0) / 255.0  # Reshape and normalize
        genre_probabilities = model.predict(img)
        predicted_genre_index = np.argmax(genre_probabilities, axis=1)[0]
        genre_votes[predicted_genre_index] = genre_votes.get(predicted_genre_index, 0) + 1

    # Determine the most common predicted genre
    most_common_genre_index = max(genre_votes, key=genre_votes.get)
    most_common_genre = genre_mapping.get(str(most_common_genre_index), "Unknown Genre")
    return most_common_genre


if __name__ == "__main__":
    model_path = 'best_model.keras'
    spectrogram_dir = 'Song_Spectrograms'  # Directory containing spectrograms
    genre_mapping = load_genre_mapping()  # Load genre mapping from JSON

    model = load_model_for_genre_prediction(model_path)
    most_common_genre = load_and_predict_genre(model, spectrogram_dir, genre_mapping)
    print(f"Most common predicted genre: {most_common_genre}")