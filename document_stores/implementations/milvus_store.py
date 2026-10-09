"""
Milvus Document Store stub - for compatibility
"""

from typing import Any, Dict, List, Optional

from haystack import Document

MILVUS_AVAILABLE = False

__all__ = ["MilvusDocumentStore", "MILVUS_AVAILABLE"]


class MilvusDocumentStore:
    """Stub implementation of MilvusDocumentStore"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        raise NotImplementedError("Milvus support not implemented in this version")

    def add_documents(self, documents: List[Document]) -> bool:
        return False
