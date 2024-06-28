import threading
import numpy as np
import sqlite3
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from llm_search.config import Config
from llm_search.embeddings import get_document_embeddings

logger = logging.getLogger(__name__)

class Searcher:
    def __init__(self, indexer_ref):
        self.executor = ThreadPoolExecutor(max_workers=1)  # Single thread executor
        self.current_future = None
        self.indexer_ref = indexer_ref
        self.lock = threading.Lock()

    def search(self, query, top_n=5):
        """Submit a search task, canceling any ongoing search."""
        with self.lock:
            if self.current_future and not self.current_future.done():
                self.current_future.cancel()
            self.current_future = self.executor.submit(self._search_task, query, top_n)
            return self.current_future

    def _search_task(self, query, top_n):
        """Perform the search for similar documents in the database."""
        future = self.current_future

        if self.indexer_ref.get_index() is None:
            logger.error("Index is not initialized.")
            return []

        query_embedding = get_document_embeddings(query)[0]
        logger.debug(f"Query embedding: {query_embedding}")

        conn = sqlite3.connect(Config.DB_PATH)
        c = conn.cursor()
        results = []
        unique_files = set()

        if top_n is not None:
            upper_bound_top_n = 5 * top_n
        else:
            c.execute("SELECT COUNT(DISTINCT file_path) FROM embeddings")
            top_n = c.fetchone()[0]
            upper_bound_top_n = top_n

        distances, indices = self.indexer_ref.get_index().search(np.array([query_embedding]), upper_bound_top_n)
        logger.debug(f"Search distances: {distances}, indices: {indices}")

        for idx in indices[0]:
            if future.cancelled():
                logger.info("Search task was cancelled.")
                conn.close()
                return []
            if idx == -1:
                logger.warning("No results found.")
                break
            idx = int(idx)
            c.execute("SELECT file_path FROM embeddings WHERE id = ?", (idx,))
            result = c.fetchone()
            if result:
                file_path = result[0]
                if file_path not in unique_files:
                    unique_files.add(file_path)
                    distance = distances[0][np.where(indices[0] == idx)[0][0]]
                    results.append((file_path, distance))
                    if len(results) == top_n:
                        break
            else:
                logger.warning(f"Document with id {idx} not found in the database.")

        conn.close()
        logger.debug(f"Search results: {results}")
        return results

    def shutdown(self):
        """Shut down the executor and cancel any ongoing tasks."""
        with self.lock:
            if self.current_future and not self.current_future.done():
                self.current_future.cancel()
            self.executor.shutdown(wait=False)
