import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image


def create_and_slice_spectrogram(file_path, output_dir='Song_Spectrograms', slice_size=128):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        y, sr = librosa.load(file_path)
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
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        full_output_filename = os.path.join(output_dir, f"{base_filename}.jpg")
        # Создание спектрограммы с заданными параметрами
        librosa.display.specshow(S_dB, cmap='gray_r')
        plt.savefig(full_output_filename, bbox_inches=None, pad_inches=0)
        plt.close()

        # Нарезка спектрограммы на меньшие части
        img = Image.open(full_output_filename)
        width, height = img.size
        for i in range(0, width, slice_size):
            for j in range(0, height, slice_size):
                if i + slice_size <= width and j + slice_size <= height:
                    img_cropped = img.crop((i, j, i + slice_size, j + slice_size))
                    output_filename = os.path.join(output_dir,
                                                   f"{base_filename}_{i // slice_size}_{j // slice_size}.jpg")
                    img_cropped.save(output_filename)

        # Удаляем исходную полную спектрограмму
        os.remove(full_output_filename)

        print(f"Processed and sliced {file_path} successfully.")
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")



if __name__ == '__main__':
    file_path = 'Dataset/fma_small/108464.mp3'
    create_and_slice_spectrogram(file_path)