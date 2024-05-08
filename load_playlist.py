import os
import logging
from keras.models import load_model
from keras.models import Model
from single_song_processor import create_and_slice_spectrogram
from load_track import parse_filename, load_genre_mapping, insert_track_data, predict_genre_and_features

logging.basicConfig(level=logging.DEBUG)


def load_models(model_path):
    full_model = load_model(model_path)
    feature_model = Model(inputs=full_model.inputs, outputs=full_model.layers[-2].output)
    return full_model, feature_model


def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        os.remove(file_path)


def process_directory(directory, output_dir, db_path, full_model, feature_model):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp3')]
    genre_mapping = load_genre_mapping()

    for file_path in files:
        logging.debug(f"Processing file: {file_path}")
        title, artist = parse_filename(file_path)
        create_and_slice_spectrogram(file_path, output_dir)
        genre_top, features = predict_genre_and_features(output_dir, full_model, feature_model, genre_mapping)
        insert_track_data(title, artist, genre_top, features, file_path, db_path)
        clear_directory(output_dir)  # Clear the directory after processing each track

        logging.info(f"Processed {title} by {artist} and data saved.")


# Основной блок исполнения
if __name__ == "__main__":
    model_path = "best_model.keras"
    full_model, feature_model = load_models(model_path)
    music_directory = 'Dataset/fma_small1'
    output_dir = 'Song_Spectrograms'
    db_path = "music_features.db"
    process_directory(music_directory, output_dir, db_path, full_model, feature_model)