import os
import pickle
import sqlite3
from typing import Callable, List, Optional
import faiss
import logging

import numpy as np

from llm_search.config import Config
from llm_search.database import create_database, get_last_modified_time, insert_or_update_embedding
from llm_search.embeddings import get_document_embeddings
from llm_search.extractor import extract_text_from_file
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class FileHandler(FileSystemEventHandler):
    def __init__(self, indexer):
        self.indexer = indexer

    def on_modified(self, event):
        if not event.is_directory:
            logger.info(f"File modified: {event.src_path}")
            self.indexer.update_index(event.src_path, force_update=True)

    def on_created(self, event):
        if not event.is_directory:
            logger.info(f"File created: {event.src_path}")
            self.indexer.update_index(event.src_path, force_update=True)

    def on_deleted(self, event):
        if not event.is_directory:
            logger.info(f"File deleted: {event.src_path}")
            self.indexer.remove_from_index(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            logger.info(f"File moved from {event.src_path} to {event.dest_path}")
            self.indexer.remove_from_index(event.src_path)
            self.indexer.update_index(event.dest_path, force_update=True)

class Indexer:
    """Class to index files in the specified directories and update the FAISS index."""
    def __init__(self, on_index_update_callback: Optional[Callable[[], None]] = None) -> None:
        self.index = None
        self.observer = Observer()
        self.on_index_update_callback = on_index_update_callback if on_index_update_callback else lambda: None

    def set_index_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set the callback function to be called when the index is updated."""
        self.on_index_update_callback = callback

    def init_index(self) -> None:
        """Initialize the FAISS index and database."""
        create_database(Config.DB_PATH)
        dimension = get_document_embeddings("test for dimension").shape[1]
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))

        conn = sqlite3.connect(Config.DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, embedding FROM embeddings ORDER BY id")
        results = c.fetchall()
        conn.close()
        if results:
            embeddings = []
            ids = []
            for result in results:
                id_, embedding_blob = result
                embedding = np.array(pickle.loads(embedding_blob))
                embeddings.append(embedding)
                ids.append(id_)
            embeddings = np.asarray(embeddings)
            ids = np.asarray(ids)
            self.index.add_with_ids(embeddings, ids)
            self.on_index_update_callback()

    def start_indexer(self) -> None:
        if self.get_index() is None:
            self.init_index()

        handler = FileHandler(self)
        for directory in Config.INDEX_DIRECTORIES:
            self.observer.schedule(handler, directory, recursive=True)
        self.observer.start()

    def stop_indexer(self) -> None:
        self.observer.stop()
        self.observer.join()

    def update_indexes(self, directories: List[str]) -> None:
        """Clear the database and re-index all files in the specified directories."""
        logger.info("Starting force update of all indexes...")

        # Clear the database
        conn = sqlite3.connect(Config.DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM embeddings")
        conn.commit()
        conn.close()

        # Clear the FAISS index
        self.index.reset()

        # Re-index all files
        for directory in directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    self.update_index(file_path, force_update=True)

        logger.info("Force update of all indexes completed.")
        self.on_index_update_callback()

    def update_index(self, file_path: str, force_update: bool = False) -> None:
        """Update the index for the specified file."""
        logger.info(f"Updating index for file: {file_path}...")

        last_modified = os.path.getmtime(file_path)
        db_last_modified = get_last_modified_time(Config.DB_PATH, file_path)
        if force_update or db_last_modified is None or last_modified > db_last_modified:
            text = extract_text_from_file(file_path)
            if text and text != "":
                embeddings = get_document_embeddings(text)
                chunk_ids = insert_or_update_embedding(Config.DB_PATH, file_path, embeddings, last_modified)

                # Update FAISS index
                conn = sqlite3.connect(Config.DB_PATH)
                c = conn.cursor()
                c.execute('SELECT id, chunk_index FROM embeddings WHERE file_path = ?', (file_path,))
                existing_chunks = c.fetchall()
                conn.close()

                existing_chunk_ids = set([chunk_id for chunk_id, _ in existing_chunks])
                new_chunk_ids = set(chunk_ids)

                # Remove old embeddings that are no longer present
                chunks_to_remove = existing_chunk_ids - new_chunk_ids
                if chunks_to_remove:
                    self.index.remove_ids(np.array(list(chunks_to_remove)))

                # Add new embeddings
                for chunk_index, embedding in enumerate(embeddings):
                    chunk_id = chunk_ids[chunk_index]
                    self.index.add_with_ids(np.array([embedding]), np.array([chunk_id]))
                self.on_index_update_callback()

        logger.info("Index update completed.")

    def remove_from_index(self, file_path: str) -> None:
        """Remove the file from both the database and FAISS index."""
        conn = sqlite3.connect(Config.DB_PATH)
        c = conn.cursor()
        c.execute('SELECT id FROM embeddings WHERE file_path = ?', (file_path,))
        chunk_ids = [row[0] for row in c.fetchall()]
        
        if chunk_ids:
            self.index.remove_ids(np.array(chunk_ids))
            c.execute('DELETE FROM embeddings WHERE file_path = ?', (file_path,))
            conn.commit()
            self.on_index_update_callback()
        
        conn.close()
        logger.info(f"Removed {file_path} from index.")

    def get_index(self) -> Optional[faiss.IndexIDMap]:
        """Get the FAISS index."""
        return self.index
