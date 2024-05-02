import sqlite3

def query_data(db_path="music_features.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM features')
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    conn.close()


if __name__ == "__main__":
    query_data()