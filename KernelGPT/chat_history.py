"""Small local SQLite store for persistent KernelGPT conversations."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from config import PROJECT_DIR

DB_PATH = PROJECT_DIR / "chat_history.sqlite3"


def _connect():
    db = sqlite3.connect(DB_PATH, timeout=10)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL,
            FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
        );
    """)
    return db


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_chats():
    with _connect() as db:
        return db.execute("SELECT id,title,updated_at FROM chats ORDER BY updated_at DESC").fetchall()


def create_chat(title="New chat"):
    chat_id, now = uuid4().hex, _now()
    with _connect() as db:
        db.execute("INSERT INTO chats VALUES(?,?,?,?)", (chat_id,title,now,now))
    return chat_id


def load_messages(chat_id):
    with _connect() as db:
        return [{"role":role,"content":content} for role,content in db.execute(
            "SELECT role,content FROM messages WHERE chat_id=? ORDER BY id", (chat_id,)
        )]


def add_message(chat_id, role, content):
    now = _now()
    with _connect() as db:
        db.execute("INSERT INTO messages(chat_id,role,content,created_at) VALUES(?,?,?,?)", (chat_id,role,content,now))
        if role == "user":
            title = content.strip().replace("\n", " ")[:52] or "New chat"
            db.execute("UPDATE chats SET title=?,updated_at=? WHERE id=? AND title='New chat'", (title,now,chat_id))
        else:
            db.execute("UPDATE chats SET updated_at=? WHERE id=?", (now,chat_id))


def delete_chat(chat_id):
    with _connect() as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
        db.execute("DELETE FROM chats WHERE id=?", (chat_id,))
