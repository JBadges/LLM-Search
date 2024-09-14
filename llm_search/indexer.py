import os
import pickle
import threading
from typing import Callable, List, Optional
import faiss
import logging

import numpy as np

from llm_search.config import Config
from llm_search.database import create_database, get_db_connection, get_last_modified_time, insert_or_update_embedding
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
        self.update_thread = None
        self.observer = Observer()
        self.on_index_update_callback = on_index_update_callback if on_index_update_callback else lambda: None
        self.stop_flag = threading.Event()

    def set_index_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set the callback function to be called when the index is updated."""
        self.on_index_update_callback = callback

    def init_index(self) -> None:
        """Initialize the FAISS index and database."""
        create_database(Config.DB_PATH)
        dimension = get_document_embeddings("test for dimension").shape[1]
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))

        with get_db_connection(Config.DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT id, embedding FROM embeddings ORDER BY id")
            results = c.fetchall()

        if not results:
            logger.info("No embeddings found in the database.")
            return
        
        try:
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
        except (pickle.UnpicklingError, ValueError, TypeError) as e:
            logger.error(f"Error processing embeddings: {e}")
        except faiss.RuntimeError as e:
            logger.error(f"Error adding embeddings to FAISS index: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during index initialization: {e}")

    def start_indexer(self) -> None:
        if self.get_index() is None:
            self.init_index()

        self.stop_flag.clear()
        self.update_thread = threading.Thread(target=self._run_initial_check)
        self.update_thread.start()

        handler = FileHandler(self)
        for directory in Config.INDEX_DIRECTORIES:
            self.observer.schedule(handler, directory, recursive=True)
        self.observer.start()

    def stop_indexer(self) -> None:
        self.stop_flag.set()
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)  # Wait up to 5 seconds for the thread to finish
            if self.update_thread.is_alive():
                logger.warning("Initial index check did not finish in time, but stop flag is set")
        self.observer.stop()
        self.observer.join()

    def _run_initial_check(self):
        try:
            logger.info("Starting initial index check and update...")
            self.check_and_update_index()
            logger.info("Initial index check and update completed.")
        except Exception as e:
            logger.error(f"Error during initial index check: {e}")

    def update_indexes(self, directories: List[str]) -> None:
        """Clear the database and re-index all files in the specified directories."""
        logger.info("Starting force update of all indexes...")

        # Clear the database
        with get_db_connection(Config.DB_PATH) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM embeddings")
            conn.commit()

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
            # Add a preamble to the text with file information
            file_name = os.path.basename(file_path)
            file_extension = os.path.splitext(file_name)[1]
            preamble = f"""{{
                "file_name": "{file_name}",
                "file_path": "{file_path}",
                "file_extension": "{file_extension}",
            }}"""
            text = preamble + "\n\n" + text
            if text and text != "":
                embeddings = get_document_embeddings(text)
                chunk_ids = insert_or_update_embedding(Config.DB_PATH, file_path, embeddings, last_modified)

                # Update FAISS index
                existing_chunks = []
                with get_db_connection(Config.DB_PATH) as conn:
                    c = conn.cursor()
                    c.execute('SELECT id, chunk_index FROM embeddings WHERE file_path = ?', (file_path,))
                    existing_chunks = c.fetchall()

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
        with get_db_connection(Config.DB_PATH) as conn:
            c = conn.cursor()
            c.execute('SELECT id FROM embeddings WHERE file_path = ?', (file_path,))
            chunk_ids = [row[0] for row in c.fetchall()]
            
            if chunk_ids:
                self.index.remove_ids(np.array(chunk_ids))
                c.execute('DELETE FROM embeddings WHERE file_path = ?', (file_path,))
                conn.commit()
                self.on_index_update_callback()
            
            logger.info(f"Removed {file_path} from index.")

    def get_index(self) -> Optional[faiss.IndexIDMap]:
        """Get the FAISS index."""
        return self.index
    
    def check_and_update_index(self) -> None:
        """Check and update the index to ensure all files are up to date."""
        logger.info("Checking and updating index...")
        
        # Get all file paths in the database
        with get_db_connection(Config.DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT DISTINCT file_path FROM embeddings")
            db_file_paths = set(row[0] for row in c.fetchall())
            
            # Get all file paths in the indexed directories
            current_file_paths = set()
            for directory in Config.INDEX_DIRECTORIES:
                for root, _, files in os.walk(directory):
                    for file in files:
                        if self.stop_flag.is_set():
                            logger.info("Stopping index check and update due to stop flag")
                            return
                        current_file_paths.add(os.path.join(root, file))
            
            # Remove entries for files that no longer exist
            files_to_remove = db_file_paths - current_file_paths
            for file_path in files_to_remove:
                if self.stop_flag.is_set():
                    logger.info("Stopping index check and update due to stop flag")
                    return
                logger.info(f"Removing index for deleted file: {file_path}")
                self.remove_from_index(file_path)
            
            # Update entries for existing files
            files_to_check = db_file_paths.intersection(current_file_paths)
            for file_path in files_to_check:
                if self.stop_flag.is_set():
                    logger.info("Stopping index check and update due to stop flag")
                    return
                last_modified = os.path.getmtime(file_path)
                db_last_modified = get_last_modified_time(Config.DB_PATH, file_path)
                if db_last_modified is None or last_modified > db_last_modified:
                    logger.info(f"Updating index for modified file: {file_path}")
                    self.update_index(file_path, force_update=False)
            
            # Add new files
            new_files = current_file_paths - db_file_paths
            for file_path in new_files:
                if self.stop_flag.is_set():
                    logger.info("Stopping index check and update due to stop flag")
                    return
                logger.info(f"Adding new file to index: {file_path}")
                self.update_index(file_path, force_update=True)
            
            logger.info("Index check and update completed.")
