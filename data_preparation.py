import os
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_spectrogram(verbose=0, mode=None):
    output_dir = 'Train_Spectrogram_Images' if mode == 'Train' else 'Test_Spectrogram_Images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    audio_path = 'Dataset/fma_small' if mode == 'Train' else 'Dataset/DLMusicTest_30'
    metadata_path = 'Dataset/fma_metadata/tracks.csv'
    tracks = pd.read_csv(metadata_path, header=[0, 1], index_col=0)
    audio_files = [os.path.join(audio_path, f) for f in os.listdir(audio_path) if f.endswith('.mp3')]

    for file in audio_files:
        try:
            track_id = int(os.path.basename(file).split('.')[0])
            genre = tracks.loc[track_id, ('track', 'genre_top')]
            y, sr = librosa.load(file)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S)

            # Изменение размера фигуры для спектрограммы
            fig_size = plt.rcParams["figure.figsize"]
            fig_size[0] = float(S_dB.shape[1]) / float(100)
            fig_size[1] = float(S_dB.shape[0]) / float(100)
            plt.rcParams["figure.figsize"] = fig_size

            plt.axis('off')
            plt.axes([0., 0., 1., 1.0], frameon=False,
                     xticks=[], yticks=[])
            output_filename = os.path.join(output_dir, f"{track_id}_{genre}.jpg")
            librosa.display.specshow(S_dB, cmap='gray_r')
            plt.savefig(output_filename, bbox_inches=None, pad_inches=0)
            plt.close()

            if verbose:
                print(f"Processed {file} successfully.")
        except Exception as e:
            print(f"Failed to process {file}: {e}")

    if verbose:
        print(f"Spectrograms created in {output_dir}")


def slice_spect(verbose=0, mode=None, slice_size=128):
    input_dir = 'Train_Spectrogram_Images' if mode == 'Train' else 'Test_Spectrogram_Images'
    output_dir = f"{mode}_Sliced_Images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')]
    processed_files = 0

    for filename in filenames:
        img = Image.open(filename)
        width, height = img.size
        for i in range(0, width, slice_size):
            for j in range(0, height, slice_size):
                if i + slice_size <= width and j + slice_size <= height:
                    img_cropped = img.crop((i, j, i + slice_size, j + slice_size))
                    output_filename = os.path.join(output_dir, f"{os.path.basename(filename).replace('.jpg', '')}_{i//slice_size}_{j//slice_size}.jpg")
                    img_cropped.save(output_filename)
                    if verbose:
                        print(f"Saved sliced image: {output_filename}")

        processed_files += 1
    if verbose:
        print(f"Sliced images saved in {output_dir}. Total processed files: {processed_files}")

if __name__ == "__main__":
    create_spectrogram(verbose=1, mode="Train")
    slice_spect(verbose=1, mode="Train")
