import numpy as np
from typing import List

def normalize_vector(vector: List[float]) -> List[float]:
    """Normalize vector to unit length"""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return (vector / norm).tolist()

def batch_process_embeddings(texts: List[str], embedder, batch_size: int = 10) -> List[List[float]]:
    """Process embeddings in batches to handle rate limits"""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedder.embed_documents(batch)
        embeddings.extend(batch_embeddings)
    return embeddings