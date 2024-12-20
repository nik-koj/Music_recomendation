import sys
import os
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QHBoxLayout, QLineEdit, QTextEdit, QSpinBox, QFormLayout, QListWidget)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCompleter
from PyQt5.QtWidgets import QTabWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QFormLayout, QListWidget, QSizePolicy, QCompleter, QMessageBox
from PyQt5.QtCore import QSize
from PyQt5.QtCore import QStringListModel
from PyQt5.QtCore import QThread, pyqtSignal
from single_song_processor import create_and_slice_spectrogram
from load_track import load_models, process_track
import logging
import shutil
from PyQt5.QtWidgets import QSlider
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'TrackRec'
        self.model_path = "best_model.keras"
        self.full_model, self.feature_model = load_models(self.model_path)
        self.player = QMediaPlayer()
        self.initUI()
        self.setup_autocomplete()
        self.player.stateChanged.connect(self.update_play_button)
        self.player.positionChanged.connect(self.update_slider_position)
        self.player.durationChanged.connect(self.update_slider_range)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #202020; color: #E0E0E0; font-size: 14px;")

        layout = QVBoxLayout()
        self.setLayout(layout)

        form_layout = QFormLayout()
        self.song_input = QLineEdit()
        self.song_input.setFixedHeight(35)

        self.recommendations_count = QSlider(Qt.Horizontal)
        self.recommendations_count.setMinimum(1)
        self.recommendations_count.setMaximum(500)
        self.recommendations_count.setTickPosition(QSlider.TicksBelow)
        self.recommendations_count.setTickInterval(1)
        self.recommendations_count.setSingleStep(1)

        self.recommendations_label = QLabel('1')
        self.recommendations_count.valueChanged.connect(self.update_label)

        recommendations_layout = QHBoxLayout()
        recommendations_layout.addWidget(self.recommendations_count)
        recommendations_layout.addWidget(self.recommendations_label)

        form_layout.addRow('Название песни:', self.song_input)
        form_layout.addRow('Количество рекомендаций:', recommendations_layout)
        layout.addLayout(form_layout)

        # Кнопки
        buttons_layout = QHBoxLayout()
        btn_recommend = QPushButton('Подобрать треки')
        btn_recommend.setFixedSize(200, 40)
        btn_recommend.clicked.connect(self.get_recommendations)
        btn_recommend.setStyleSheet("background-color: #1DB954; color: #FFFFFF; font-weight: bold;")
        buttons_layout.addWidget(btn_recommend)

        btn_load = QPushButton('Загрузить трек')
        btn_load.setFixedSize(200, 40)
        btn_load.clicked.connect(self.openFileNameDialog)
        btn_load.setStyleSheet("background-color: #1DB954; color: #FFFFFF; font-weight: bold;")
        buttons_layout.addWidget(btn_load)

        btn_load_playlist = QPushButton('Загрузить Плейлист')
        btn_load_playlist.setFixedSize(200, 40)
        btn_load_playlist.clicked.connect(self.load_playlist)
        btn_load_playlist.setStyleSheet("background-color: #1DB954; color: #FFFFFF; font-weight: bold;")
        buttons_layout.addWidget(btn_load_playlist)

        layout.addLayout(buttons_layout)

        # Лист рекомендаций
        self.recommendations_list = QListWidget()
        self.recommendations_list.itemDoubleClicked.connect(self.play_selected_track)
        layout.addWidget(self.recommendations_list)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.canvas.hide()

        self.currentTrackLabel = QLabel("Сейчас играет: ")
        layout.addWidget(self.currentTrackLabel)

        #Плеер
        controls = QHBoxLayout()

        self.playButton = QPushButton()
        self.playButton.setIcon(QIcon('icons/play.png'))
        self.playButton.setIconSize(QSize(30, 30))
        self.playButton.clicked.connect(self.toggle_play)
        controls.addWidget(self.playButton)


        # Slider for track progress
        #sliderLayout = QHBoxLayout()
        self.currentTimeLabel = QLabel("00:00")
        self.totalTimeLabel = QLabel("00:00")
        self.trackSlider = QSlider(Qt.Horizontal)
        self.trackSlider.sliderMoved.connect(self.set_position)
        controls.addWidget(self.currentTimeLabel)
        controls.addWidget(self.trackSlider)
        controls.addWidget(self.totalTimeLabel)
        #layout.addLayout(sliderLayout)
        layout.addLayout(controls)




        self.show()

    def update_label(self, value):
        self.recommendations_label.setText(str(value))

    def play_selected_track(self, item):
        track_info = item.text()
        title = track_info.split(" - ")[0]
        self.load_and_play(title)

    def load_and_play(self, title):
        conn = sqlite3.connect("music_features.db")
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM features WHERE title=?", (title,))
        result = cursor.fetchone()
        conn.close()
        if result:
            file_path = result[0]
            self.currentTrackLabel.setText(f"Сейчас играет: {title}")
            self.play_music(file_path)
        else:
            QMessageBox.warning(self, "Playback Error", "Could not find the track in the database.")


    def play_music(self, file_path):
        url = QUrl.fromLocalFile(file_path)
        content = QMediaContent(url)
        self.player.setMedia(content)
        self.player.play()


    def toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            if self.player.mediaStatus() == QMediaPlayer.NoMedia and self.song_input.text():
                self.load_and_play(self.song_input.text())
            else:
                self.player.play()

    def update_play_button(self, state):
        if state == QMediaPlayer.PlayingState:
            self.playButton.setIcon(QIcon('icons/pause.png'))
        else:
            self.playButton.setIcon(QIcon('icons/play.png'))

    def update_slider_position(self, position):
        self.trackSlider.setValue(position)
        self.currentTimeLabel.setText(self.format_time(position))

    def update_slider_range(self, duration):
        self.trackSlider.setRange(0, duration)
        self.totalTimeLabel.setText(self.format_time(duration))

    def set_position(self, position):
        self.player.setPosition(position)

    def format_time(self, ms):
        seconds = round(ms / 1000)
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02}:{seconds:02}"


    def openFileNameDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Выберите аудиофайл", "", "Audio Files (*.mp3 *.wav)",
                                                  options=options)
        if fileName:
            self.process_and_display_song(fileName)
            self.setup_autocomplete()
            #self.play_music(fileName)

    def clear_directory(self, directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


    def process_and_display_song(self, file_path):
        output_dir = 'Song_Spectrograms'
        db_path = "music_features.db"

        # Обработка трека и получение информации
        title, artist, genre_top, features = process_track(file_path, output_dir, db_path, self.full_model,
                                                           self.feature_model)

        # Отображение полученной информации
        self.song_input.setText(f"{title} - {artist}")

        # Вывод жанра во всплывающем окне с форматированием текста
        QMessageBox.information(self, "Предсказанный Жанр",
                                f"Преобладающий жанр в треке {title} - {artist}: {genre_top}")

        # Очистка директории Song_Spectrograms
        self.clear_directory(output_dir)
        # Обновление списка доступных треков для автозаполнения
        self.setup_autocomplete()


    def load_playlist(self):
        # Выбор директории с музыкальными файлами
        directory = QFileDialog.getExistingDirectory(self, "Выберите папку с треками")
        if directory:
            self.process_directory(directory)

    def process_directory(self, directory):
        # Список для сбора информации о загруженных треках
        processed_tracks_info = []
        for filename in os.listdir(directory):
            if filename.endswith('.mp3'):
                file_path = os.path.join(directory, filename)
                title, artist, genre_top, features = process_track(file_path, 'Song_Spectrograms', 'music_features.db',
                                                                   self.full_model, self.feature_model)
                # Собираем информацию о каждом треке
                processed_tracks_info.append(f"{title} - {artist}: Жанр - {genre_top}")
                # Очистка директории
                self.clear_directory('Song_Spectrograms')

        # Показываем информационное сообщение с результатами обработки
        QMessageBox.information(self, "Обработка завершена", "\n".join(processed_tracks_info))
        self.setup_autocomplete()



    def get_features_for_track(self, track_title):
        conn = sqlite3.connect("music_features.db")
        cursor = conn.cursor()
        cursor.execute("SELECT features FROM features WHERE title=?", (track_title,))
        row = cursor.fetchone()
        conn.close()
        return np.array(row[0].split(','), dtype=float) if row else None

    def get_all_tracks_features(self):
        conn = sqlite3.connect("music_features.db")
        cursor = conn.cursor()
        cursor.execute("SELECT title, artist, genre_top, features FROM features")
        rows = cursor.fetchall()
        conn.close()
        return [{'title': row[0], 'artist': row[1], 'genre_top': row[2],
                 'features': np.array(row[3].split(','), dtype=float)} for row in rows]


    def get_recommendations(self):
        selected_track = self.song_input.text().split(" - ")[0]
        self.load_and_play(selected_track)
        conn = sqlite3.connect("music_features.db")
        cursor = conn.cursor()
        cursor.execute("SELECT features FROM features WHERE title=?", (selected_track,))
        result = cursor.fetchone()
        if not result:
            print("Трек не найден")
            return

        selected_features = np.array(result[0].split(','), dtype=float).reshape(1, -1)
        selected_features = selected_features / np.linalg.norm(selected_features) if np.linalg.norm(
            selected_features) > 0 else selected_features

        cursor.execute("SELECT title, artist, genre_top, features FROM features")
        all_tracks = cursor.fetchall()

        similarities = []
        for track in all_tracks:
            if track[0] == selected_track:
                continue  # Пропустить трек, если он совпадает с выбранным
            features = np.array(track[3].split(','), dtype=float).reshape(1, -1)
            norm = np.linalg.norm(features)
            features = features / norm if norm > 0 else features
            sim_score = cosine_similarity(selected_features, features)[0][0]
            similarities.append((track[0], track[1], track[2], sim_score))

        similarities.sort(key=lambda x: x[3], reverse=True)
        top_n = int(self.recommendations_count.value())

        self.recommendations_list.clear()
        for track in similarities[:top_n]:
            self.recommendations_list.addItem(f"{track[0]} - {track[1]} - {track[2]} - Сходство: {track[3]:.10f}")

        conn.close()

    def load_track_titles(self):
        conn = sqlite3.connect("music_features.db")
        cursor = conn.cursor()
        cursor.execute('SELECT title, artist FROM features')
        rows = cursor.fetchall()
        conn.close()
        return [f"{row[0]} - {row[1]}" for row in rows]

    def setup_autocomplete(self):
        titles = self.load_track_titles()
        completer = QCompleter(titles, self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        self.song_input.setCompleter(completer)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())