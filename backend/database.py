# backend/database.py
import sqlite3
import os

def get_connection():
    os.makedirs("data", exist_ok=True)  # Ensures the folder exists
    conn = sqlite3.connect("data/data.db", check_same_thread=False)
    return conn

def create_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            filetype TEXT,
            content TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_file(filename, filetype, content):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO uploads (filename, filetype, content) VALUES (?, ?, ?)",
                   (filename, filetype, content))
    conn.commit()
    conn.close()
