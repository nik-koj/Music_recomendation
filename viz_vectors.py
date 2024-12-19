import matplotlib.pyplot as plt
import sqlite3
import numpy as np
import pandas as pd


def fetch_data_from_database(db_path="music_features.db"):
    conn = sqlite3.connect(db_path)
    query = "SELECT title, artist, genre_top, features FROM features"
    data = pd.read_sql_query(query, conn)
    conn.close()

    # Преобразуем строку признаков обратно в numpy array
    data['features'] = data['features'].apply(lambda x: np.array(list(map(float, x.split(',')))))
    return data


def visualize_features(data, genres, num_tracks=1):
    # Фильтрация данных по жанрам и случайный выбор одного трека для каждого жанра
    filtered_data = pd.DataFrame()
    for genre in genres:
        genre_data = data[data['genre_top'] == genre]
        if not genre_data.empty:
            genre_data = genre_data.sample(n=num_tracks)
            filtered_data = pd.concat([filtered_data, genre_data])

    # Определение цветов для каждого жанра
    colors = plt.cm.get_cmap('tab10', len(genres))

    # Построение графика
    plt.figure(figsize=(14, 8))  # Размер фигуры

    for i, (idx, row) in enumerate(filtered_data.iterrows()):
        features = row['features']
        color = colors(i % len(genres))  # Цвет для текущего жанра
        plt.plot(range(len(features)), features, linestyle='-', color=color, linewidth=0.7, alpha=0.7,
                 label=f"{row['genre_top']}: {row['title']} - {row['artist']}")

    # Создаем список для легенды
    legend_labels = [f"{row['genre_top']}: {row['title']} - {row['artist']}" for idx, row in filtered_data.iterrows()]
    legend_colors = [colors(i % len(genres)) for i in range(len(filtered_data))]

    # Параметры графика
    plt.title("Визуализация признаков для каждого трека")
    plt.xlabel("Компоненты вектора признаков")
    plt.ylabel("Значения компонентов")
    plt.grid(True)

    # Создаем легенду отдельно и размещаем её сверху
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=legend_colors[i], lw=2, label=legend_labels[i]) for i in range(len(legend_labels))]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.2),
               ncol=2, fancybox=True, shadow=True)  # Увеличиваем y-координату для большего пространства

    plt.tight_layout(rect=[0, 0, 1, 0.85])  # Оставляем больше места сверху для легенды
    plt.show()


if __name__ == "__main__":
    genres = ["Электронный", "Экспериментальный", "Фолк", "Хип-хоп", "Инструментальный", "Международный", "Поп", "Рок"]
    data = fetch_data_from_database()
    visualize_features(data, genres)
