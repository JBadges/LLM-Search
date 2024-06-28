import os
import pickle
import sqlite3
import threading
import faiss
import logging

import numpy as np

from llm_search.config import Config
from llm_search.database import create_database, get_last_modified_time, insert_or_update_embedding
from llm_search.embeddings import get_document_embeddings
from llm_search.extractor import extract_text_from_file, is_text_file

logger = logging.getLogger(__name__)

class Indexer:
    def __init__(self):
        self.index = None
        self.indexer_thread = None
        self.stop_event = threading.Event()

    def init_index(self):
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

    def start_indexer_thread(self):
        """Start the indexer thread."""
        if self.indexer_thread is None or not self.indexer_thread.is_alive():
            self.indexer_thread = threading.Thread(target=self.update_directories_index, args=(Config.INDEX_DIRECTORIES,))
            self.indexer_thread.start()

    def stop_indexer_thread(self):
        """Stop the indexer thread."""
        if self.indexer_thread is not None and self.indexer_thread.is_alive():
            self.stop_event.set()
            self.indexer_thread.join()

    def update_directories_index(self, dirs):
        """Update the index for the specified directories."""
        if self.get_index() is None:
            self.init_index()

        while not self.stop_event.is_set():
            for directory in dirs:
                if self.stop_event.is_set():
                    break
                self.update_index(directory)

            # Sleep in intervals, checking the stop_event to wake up early if needed
            sleep_interval = 1  # Check the event every 1 second
            total_sleep_time = Config.RE_INDEX_INTERVAL_SECONDS
            elapsed_time = 0

            while elapsed_time < total_sleep_time and not self.stop_event.is_set():
                remaining_time = min(sleep_interval, total_sleep_time - elapsed_time)
                self.stop_event.wait(remaining_time)
                elapsed_time += sleep_interval

    def update_index(self, directory, force_update=False):
        logger.info(f"Updating index for directory: {directory}...")
        for root, _, files in os.walk(directory):
            if self.stop_event.is_set():
                break
            for file in files:
                if self.stop_event.is_set():
                    break

                file_path = os.path.join(root, file)

                last_modified = os.path.getmtime(file_path)
                db_last_modified = get_last_modified_time(Config.DB_PATH, file_path)
                if force_update or db_last_modified is None or last_modified > db_last_modified:
                    text = extract_text_from_file(file_path)
                    if text and text != "":
                        embeddings = get_document_embeddings(text)
                        chunk_ids = insert_or_update_embedding(Config.DB_PATH, file_path, embeddings, last_modified)

                        # Update FAISS index
                        # Get all existing chunk indices
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

        logger.info("Index update completed.")

    def get_index(self):
        """Get the FAISS index."""
        return self.index
    