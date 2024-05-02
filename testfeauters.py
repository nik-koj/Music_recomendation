import sqlite3
import numpy as np

def load_features_from_db(db_path="music_features.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT features FROM features")
    rows = cursor.fetchall()
    conn.close()
    features = [np.array(row[0].split(','), dtype=float) for row in rows]
    return features

def normalize_features(features):
    norm = np.linalg.norm(features)
    if norm == 0:
        return features
    return features / norm

# Загрузка и нормализация признаков
features = load_features_from_db()
normalized_features = [normalize_features(feat) for feat in features]

# Проверка уникальности признаков
unique_features = np.unique(np.array(normalized_features), axis=0)
print(f"Total features: {len(features)}, Unique normalized features: {len(unique_features)}")

# Проверка нормализации
for feat in normalized_features:
    norm = np.linalg.norm(feat)
    if not np.isclose(norm, 1):
        print("Error: Feature vector norm is not 1:", norm)

# Дополнительно: проверка различия между векторами
differences = [np.linalg.norm(f1 - f2) for i, f1 in enumerate(normalized_features) for f2 in normalized_features[i+1:]]
print("Average difference between feature vectors:", np.mean(differences))
print("Minimum difference between feature vectors:", np.min(differences))