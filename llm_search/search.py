import numpy as np
import sqlite3
import logging
from llm_search.embeddings import get_document_embeddings

logger = logging.getLogger(__name__)

def search(index, query, db_path, top_n=5):
    query_embedding = get_document_embeddings(query)[0]

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    results = []
    unique_files = set()

    if top_n is not None:
        upper_bound_top_n = 5 * top_n
    else:
        c.execute("SELECT COUNT(DISTINCT file_path) FROM embeddings")
        top_n = c.fetchone()[0]
        upper_bound_top_n = top_n

    distances, indices = index.search(np.array([query_embedding]), upper_bound_top_n)

    for idx in indices[0]:
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

    conn.close()
    return results
