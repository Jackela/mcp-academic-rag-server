"""
Milvus Document Store stub - for compatibility
"""

MILVUS_AVAILABLE = False

__all__ = ['MilvusDocumentStore', 'MILVUS_AVAILABLE']


class MilvusDocumentStore:
    """Stub implementation of MilvusDocumentStore"""
    
    def __init__(self, config=None):
        self.config = config or {}
        raise NotImplementedError("Milvus support not implemented in this version")
    
    def add_documents(self, documents):
        return False