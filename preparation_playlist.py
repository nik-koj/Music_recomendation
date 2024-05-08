import os
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_spectrogram(audio_path, output_dir = 'pList_Spectrograms', verbose=0):
    plt.cla()  # Очистка текущих осей
    plt.clf()  # Очистка текущей фигуры
    plt.close('all')  # Закрыть все фигуры
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    audio_files = [os.path.join(audio_path, f) for f in os.listdir(audio_path) if f.endswith('.mp3')]

    for file in audio_files:
        try:
            base_name = os.path.splitext(os.path.basename(file))[0]
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
            output_filename = os.path.join(output_dir, base_name + '.jpg')
            librosa.display.specshow(S_dB, cmap='gray_r')
            plt.savefig(output_filename, bbox_inches=None, pad_inches=0)
            plt.close()

            if verbose:
                print(f"Processed {file} successfully.")
        except Exception as e:
            print(f"Failed to process {file}: {e}")


def slice_spect(input_dir = 'pList_Spectrograms', output_dir = 'pList_SlicedSp', slice_size=128, verbose=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')]

    for filename in filenames:
        img = Image.open(filename)
        width, height = img.size
        base_name = os.path.splitext(os.path.basename(filename))[0]
        for i in range(0, width, slice_size):
            for j in range(0, height, slice_size):
                if i + slice_size <= width and j + slice_size <= height:  # Убедиться, что срез не выходит за границы изображения
                    img_cropped = img.crop((i, j, i + slice_size, j + slice_size))
                    output_filename = os.path.join(output_dir, f"{base_name}_{i // slice_size}_{j // slice_size}.jpg")
                    img_cropped.save(output_filename)
                    if verbose:
                        print(f"Saved sliced image: {output_filename}")



if __name__ == "__main__":
    user_audio_path = 'Dataset/fma_small1'
    spectrogram_dir = 'pList_Spectrograms'
    sliced_spectrogram_dir = 'pList_SlicedSp'
    create_spectrogram(user_audio_path, spectrogram_dir, verbose=1)
    slice_spect(spectrogram_dir, sliced_spectrogram_dir, verbose=1)