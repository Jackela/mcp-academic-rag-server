"""
Document Class - Core Data Structure for Academic Documents

This module provides the Document class for storing and managing basic document
information, processing status, metadata, and results from various processing stages.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


class Document:
    """
    Document class - core data structure for academic documents and metadata.

    This class stores basic document information, processing status, metadata,
    and results from various processing stages. It provides methods for updating
    document status, adding/modifying metadata, and retrieving document information.
    """

    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize Document object.

        Args:
            file_path: Path to the document file
        """
        self.document_id: str = str(uuid.uuid4())
        self.file_path: Path = Path(file_path) if isinstance(file_path, str) else file_path
        self.file_name: str = self.file_path.name
        self.file_type: str = self.file_path.suffix.lower()
        self.creation_time: datetime = datetime.now()
        self.modification_time: datetime = self.creation_time
        self.status: str = "new"  # new, processing, completed, error
        self.metadata: Dict[str, Any] = {}
        self.tags: List[str] = []
        self.content: Dict[str, Any] = {}  # Store results from processing stages
        self.processing_history: List[Dict[str, Any]] = []

    def update_status(self, status: str) -> None:
        """
        更新文档处理状态。

        Args:
            status: 新的处理状态
        """
        self.status = status
        self.modification_time = datetime.now()
        self.processing_history.append({"time": self.modification_time, "status": status})

    def add_metadata(self, key: str, value: Any) -> None:
        """
        添加或更新文档元数据。

        Args:
            key: 元数据键
            value: 元数据值
        """
        self.metadata[key] = value
        self.modification_time = datetime.now()

    def add_tag(self, tag: str) -> None:
        """
        为文档添加标签。

        Args:
            tag: 要添加的标签
        """
        if tag not in self.tags:
            self.tags.append(tag)
            self.modification_time = datetime.now()

    def store_content(self, stage: str, content: Any) -> None:
        """
        存储特定处理阶段的内容。

        Args:
            stage: 处理阶段名称
            content: 处理结果内容
        """
        self.content[stage] = content
        self.modification_time = datetime.now()

    def get_content(self, stage: str) -> Optional[Any]:
        """
        获取特定处理阶段的内容。

        Args:
            stage: 处理阶段名称

        Returns:
            该阶段的处理内容，如不存在则返回None
        """
        return self.content.get(stage)

    def to_dict(self) -> Dict:
        """
        将文档对象转换为字典表示，便于序列化。

        Returns:
            包含文档所有属性的字典
        """
        return {
            "document_id": self.document_id,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "creation_time": self.creation_time.isoformat(),
            "modification_time": self.modification_time.isoformat(),
            "status": self.status,
            "metadata": self.metadata,
            "tags": self.tags,
            "content_stages": list(self.content.keys()),
            "processing_history": self.processing_history,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Document":
        """
        从字典创建Document对象，用于反序列化。

        Args:
            data: 包含文档属性的字典

        Returns:
            创建的Document对象
        """
        doc = cls(data["file_path"])
        doc.document_id = data["document_id"]
        doc.file_name = data["file_name"]
        doc.file_type = data["file_type"]
        doc.creation_time = datetime.fromisoformat(data["creation_time"])
        doc.modification_time = datetime.fromisoformat(data["modification_time"])
        doc.status = data["status"]
        doc.metadata = data["metadata"]
        doc.tags = data["tags"]
        doc.processing_history = data["processing_history"]

        # 注意：content不包含在转换中，需要单独处理
        return doc
