import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_features_from_database(db_path="music_features.db"):
    conn = sqlite3.connect(db_path)
    query = "SELECT title, artist, genre_top, features FROM features"
    df = pd.read_sql(query, conn)
    conn.close()

    # Разделяем строку features на отдельные значения и преобразуем в массив numpy
    df['features'] = df['features'].apply(lambda x: np.array(list(map(float, x.split(',')))))

    return df

def visualize_features(df):
    features = np.stack(df['features'].values)
    genres = df['genre_top'].values

    # t-SNE проекция
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    # Визуализация
    plt.figure(figsize=(16, 10))
    unique_genres = list(set(genres))
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_genres)))

    for i, genre in enumerate(unique_genres):
        genre_indices = np.where(genres == genre)
        plt.scatter(tsne_results[genre_indices, 0], tsne_results[genre_indices, 1], color=colors[i], label=genre)

    plt.title('t-SNE визуализация музыкальных характеристик')
    plt.xlabel('Измерение 1')
    plt.ylabel('Измерение 2')
    plt.legend()
    plt.show()

if __name__ == "__main__":  # Исправлено положение этого блока
    feature_df = load_features_from_database()
    visualize_features(feature_df)