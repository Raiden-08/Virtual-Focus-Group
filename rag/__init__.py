"""
RC-TGAD RAG module — Person 2's implementation.

Public API:
    from rag.vector_store import VectorStore
    from rag.hardness import compute_h_temp, compute_h_struct, compute_h_rag
    from rag.rag_scorer import score_hardness, score_dataset
"""

from rag.vector_store import VectorStore
from rag.hardness import compute_h_temp, compute_h_struct, compute_h_rag
from rag.rag_scorer import score_hardness, score_dataset

__all__ = [
    "VectorStore",
    "compute_h_temp",
    "compute_h_struct",
    "compute_h_rag",
    "score_hardness",
    "score_dataset",
]