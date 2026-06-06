import numpy as np
import hnswlib

from similarity.config import settings


class SimilarityIndex:
    """
    Wraps an hnswlib HNSW index for approximate nearest neighbor search.

    The index uses cosine distance. All vectors should be L2-normalized
    before being added (then cosine similarity equals the dot product,
    which is what hnswlib's cosine space computes efficiently).
    """

    def __init__(self, dim: int, max_elements: int):
        self.dim = dim
        self.max_elements = max_elements
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(
            max_elements=max_elements,
            M=settings.HNSW_M,
            ef_construction=settings.HNSW_EF_CONSTRUCTION,
            random_seed=42
        )
        self.index.set_ef(settings.HNSW_EF_QUERY)

    def build(self, feature_matrix: np.ndarray) -> None:
        """Add all product vectors to the index. Row index == HNSW label."""
        row_indices = list(range(len(feature_matrix)))
        self.index.add_items(feature_matrix, ids=row_indices)

    def query(self, vector: np.ndarray, k: int, exclude_index: int) -> list:
        """
        Find k nearest neighbors of vector, excluding exclude_index.

        We fetch k+1 to ensure we have k results after removing the
        query product itself (which HNSW may or may not return as top-1).
        """
        query_vector = vector.reshape(1, -1)
        labels, _distances = self.index.knn_query(query_vector, k=k + 1)

        neighbor_indices = []
        for label in labels[0]:
            row_index = int(label)
            if row_index != exclude_index:
                neighbor_indices.append(row_index)

        return neighbor_indices[:k]

    def save(self, path: str) -> None:
        """Persist the HNSW graph to disk."""
        self.index.save_index(path)

    @classmethod
    def load(cls, path: str, dim: int, max_elements: int) -> "SimilarityIndex":
        """Reload a previously saved index from disk. Skips the build step."""
        obj = cls.__new__(cls)
        obj.dim = dim
        obj.max_elements = max_elements
        obj.index = hnswlib.Index(space="cosine", dim=dim)
        obj.index.load_index(path, max_elements=max_elements)
        obj.index.set_ef(settings.HNSW_EF_QUERY)
        return obj
