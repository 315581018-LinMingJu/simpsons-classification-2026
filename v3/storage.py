"""本機推論快取與輸出檔保護。"""

import csv
import hashlib
import io
import json
import os
import sqlite3
from pathlib import Path
from tempfile import NamedTemporaryFile

from v3.inference import Prediction


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_signature(paths):
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(json_bytes([str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]))
    return digest.hexdigest()


class Cache:
    def __init__(self, connection, metadata):
        self.connection = connection
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS predictions (
                method TEXT, id TEXT, label INTEGER, confidence REAL, disagreement REAL, seconds REAL,
                PRIMARY KEY (method, id)
            );
        """)
        encoded = json_bytes(metadata).decode()
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO settings VALUES (1, ?)", (encoded,))
            saved, = self.connection.execute("SELECT value FROM settings WHERE id = 1").fetchone()
            if saved != encoded:
                raise ValueError("Cache inputs or settings changed; choose a new --output directory")

    def read(self, method):
        rows = self.connection.execute(
            "SELECT id, label, confidence, disagreement, seconds FROM predictions WHERE method = ?",
            (method,),
        )
        return {item: Prediction(label, confidence, disagreement, seconds)
                for item, label, confidence, disagreement, seconds in rows}

    def add(self, method, batch):
        with self.connection:
            self.connection.executemany(
                "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
                [(method, item, pred.label, pred.confidence, pred.view_disagreement, pred.seconds)
                 for item, pred in batch],
            )


def publish(path, contents):
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(contents)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != contents:
                raise FileExistsError(f"Output differs from existing file: {path}; use a new --output")
            print(f"沿用 {path}", flush=True)
    finally:
        if temporary is not None:
            temporary.unlink()


def publish_csv(path, fields, rows):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(fields)
    writer.writerows(rows)
    publish(path, buffer.getvalue().encode("utf-8"))
