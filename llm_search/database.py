from contextlib import contextmanager
import sqlite3
import pickle
import logging
from typing import Generator

import torch

logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Get a connection to the database."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

def create_database(db_path: str) -> None:
    """Create the database if it does not exist."""
    with get_db_connection(db_path) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS embeddings (
                        id INTEGER PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        embedding BLOB NOT NULL,
                        last_modified REAL NOT NULL,
                        UNIQUE(file_path, chunk_index))''')
        conn.commit()

def insert_or_update_embedding(db_path: str, file_path: str, embeddings: torch.Tensor, last_modified: float) -> list[int]:
    """Insert or update the embeddings for a file in the database. Return the chunk IDs."""
    with get_db_connection(db_path) as conn:
        c = conn.cursor()
    
        c.execute('SELECT chunk_index FROM embeddings WHERE file_path = ?', (file_path,))
        existing_chunks = set(row[0] for row in c.fetchall())
        
        new_chunks = set(range(len(embeddings)))
        chunks_to_delete = existing_chunks - new_chunks

        for chunk_index in chunks_to_delete:
            c.execute('DELETE FROM embeddings WHERE file_path = ? AND chunk_index = ?', (file_path, chunk_index))
        
        chunk_ids = []
        for chunk_index, embedding in enumerate(embeddings):
            embedding_blob = pickle.dumps(embedding)
            c.execute('''INSERT OR REPLACE INTO embeddings (file_path, chunk_index, embedding, last_modified)
                        VALUES (?, ?, ?, ?)''', (file_path, chunk_index, embedding_blob, last_modified))
            chunk_id = c.lastrowid
            chunk_ids.append(chunk_id)

        conn.commit()
        logger.info(f"Updated embeddings for {file_path} in database.")
        return chunk_ids

def get_last_modified_time(db_path: str, file_path: str) -> float:
    """Get the last modified time for the specified file in the database."""
    with get_db_connection(db_path) as conn:
        c = conn.cursor()
        c.execute('SELECT MAX(last_modified) FROM embeddings WHERE file_path = ?', (file_path,))
        result = c.fetchone()
        return result[0] if result else None

def load_embeddings(db_path: str) -> list[tuple[int, str, int, torch.Tensor]]:
    """Load all embeddings from the database."""
    with get_db_connection(db_path) as conn:
        c = conn.cursor()
        c.execute('SELECT id, file_path, chunk_index, embedding FROM embeddings')
        results = c.fetchall()
        return [(row[0], row[1], row[2], pickle.loads(row[3])) for row in results]
