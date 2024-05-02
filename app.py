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
from PyQt5.QtCore import QStringListModel
from PyQt5.QtCore import QThread, pyqtSignal
from single_song_processor import create_and_slice_spectrogram
from load_track import  load_models, process_track
import logging


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Music Recommender System'
        self.model_path = "best_model.keras"
        self.full_model, self.feature_model = load_models(self.model_path)
        self.initUI()
        self.setup_autocomplete()  # Настройка автодополнения

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 800, 600)  # Установка размеров окна
        self.setStyleSheet("background-color: #282828; color: #E0E0E0; font-size: 14px;")

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Форма для ввода названия песни и выбора количества рекомендаций
        form_layout = QFormLayout()
        self.song_input = QLineEdit()
        self.recommendations_count = QSpinBox()
        self.recommendations_count.setMinimum(1)
        self.recommendations_count.setMaximum(10)
        form_layout.addRow('Song name:', self.song_input)
        form_layout.addRow('Number of recommendations:', self.recommendations_count)
        layout.addLayout(form_layout)

        # Кнопки
        btn_recommend = QPushButton('Get Recommendations')
        btn_recommend.clicked.connect(self.get_recommendations)
        btn_recommend.setStyleSheet("background-color: #1DB954; color: #FFFFFF;")
        layout.addWidget(btn_recommend)

        self.recommendations_list = QListWidget()
        layout.addWidget(self.recommendations_list)

        btn_load = QPushButton('Load Song')
        btn_load.clicked.connect(self.openFileNameDialog)
        btn_load.setStyleSheet("background-color: #1DB954; color: #FFFFFF;")
        layout.addWidget(btn_load)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.canvas.hide()

        btn_load_playlist = QPushButton('Load Playlist')
        btn_load_playlist.clicked.connect(self.load_playlist)
        layout.addWidget(btn_load_playlist)

        self.show()


    def openFileNameDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select an audio file", "", "Audio Files (*.mp3 *.wav)",
                                                  options=options)
        if fileName:
            self.process_and_display_song(fileName)

    def process_and_display_song(self, file_path):
        output_dir = 'Song_Spectrograms'
        db_path = "music_features.db"

        # Обработка трека и получение жанра
        title, artist, genre_top, features = process_track(file_path, output_dir, db_path, self.full_model,
                                                           self.feature_model)

        # Отображение полученной информации
        self.song_input.setText(f"{title} - {artist}")
        self.genre_label.setText(f"Predicted Genre: {genre_top}")

        # Обновление списка доступных треков для автозаполнения
        self.setup_autocomplete()


    def load_playlist(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select a playlist file", "", "Playlist Files (*.csv *.txt)",
                                                  options=options)
        if fileName:
            self.process_playlist(fileName)

    def process_playlist(self, file_path):
        with open(file_path, 'r') as file:
            tracks = file.readlines()
            # Обработка треков плейлиста
            for track in tracks:
                print(track.strip())  # Пример вывода пути к треку



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
        conn = sqlite3.connect("music_features.db")
        cursor = conn.cursor()
        cursor.execute("SELECT features FROM features WHERE title=?", (selected_track,))
        result = cursor.fetchone()
        if not result:
            print("Track not found")
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
            self.recommendations_list.addItem(f"{track[0]} - {track[1]} - {track[2]} - Similarity: {track[3]:.10f}")

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