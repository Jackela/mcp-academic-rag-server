"""
向量存储迁移工具测试

测试VectorStoreMigrator的迁移、备份、恢复和验证功能。
"""

import pytest
import tempfile
import shutil
import os
import json
import hashlib
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from haystack import Document as HaystackDocument

from document_stores.migration.vector_migration import (
    VectorStoreMigrator,
    migrate_vector_storage,
    backup_vector_storage
)
from document_stores.implementations.base_vector_store import BaseVectorStore


class MockVectorStore(BaseVectorStore):
    """测试用的模拟向量存储"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.documents = {}
        self.embeddings = {}
        self.next_id = 1
        self.save_calls = []
        self.load_calls = []
        
    def initialize(self) -> bool:
        self.is_initialized = True
        return True
    
    def add_documents(self, documents: List[HaystackDocument], embeddings=None) -> bool:
        if not self.is_initialized:
            return False
            
        for i, doc in enumerate(documents):
            doc_id = doc.id or f"doc_{self.next_id}"
            doc.id = doc_id
            self.documents[doc_id] = doc
            
            if embeddings and i < len(embeddings):
                self.embeddings[doc_id] = embeddings[i]
            elif doc.embedding:
                self.embeddings[doc_id] = doc.embedding
            
            self.next_id += 1
        
        return True
    
    def search(self, query_embedding, top_k=5, filters=None):
        return []
    
    def get_document_by_id(self, doc_id: str):
        return self.documents.get(doc_id)
    
    def update_document(self, doc_id: str, document, embedding=None) -> bool:
        return False
    
    def delete_document(self, doc_id: str) -> bool:
        return False
    
    def delete_all_documents(self) -> bool:
        self.documents.clear()
        self.embeddings.clear()
        return True
    
    def get_document_count(self) -> int:
        return len(self.documents)
    
    def save_index(self, path: str) -> bool:
        self.save_calls.append(path)
        
        # 模拟保存索引文件
        index_file = os.path.join(path, "mock_index.bin")
        with open(index_file, 'w') as f:
            f.write("mock index data")
        
        return True
    
    def load_index(self, path: str) -> bool:
        self.load_calls.append(path)
        
        # 检查索引文件是否存在
        index_file = os.path.join(path, "mock_index.bin")
        return os.path.exists(index_file)
    
    def get_haystack_store(self):
        """模拟Haystack存储接口"""
        mock_store = Mock()
        
        # 创建包含所有文档的列表
        all_docs = list(self.documents.values())
        mock_store.get_all_documents.return_value = all_docs
        
        return mock_store


class TestVectorStoreMigrator:
    """向量存储迁移器测试"""
    
    @pytest.fixture
    def temp_backup_dir(self):
        """创建临时备份目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_documents(self):
        """创建示例文档"""
        docs = [
            HaystackDocument(
                content="Document about AI",
                meta={"category": "technology"},
                id="doc1",
                embedding=[1.0, 0.0, 0.0]
            ),
            HaystackDocument(
                content="Document about ML",
                meta={"category": "science"},
                id="doc2", 
                embedding=[0.0, 1.0, 0.0]
            ),
            HaystackDocument(
                content="Document about DL",
                meta={"category": "research"},
                id="doc3",
                embedding=[0.0, 0.0, 1.0]
            )
        ]
        return docs
    
    @pytest.fixture
    def source_store_with_data(self, sample_documents):
        """创建包含数据的源存储"""
        config = {"type": "mock", "vector_dimension": 3}
        store = MockVectorStore(config)
        store.initialize()
        store.add_documents(sample_documents)
        return store
    
    @pytest.fixture
    def empty_target_store(self):
        """创建空的目标存储"""
        config = {"type": "mock", "vector_dimension": 3}
        store = MockVectorStore(config)
        store.initialize()
        return store
    
    def test_migrator_initialization(self):
        """测试迁移器初始化"""
        migrator = VectorStoreMigrator()
        
        assert migrator.logger is not None
        assert hasattr(migrator, 'logger')
    
    @patch('utils.vector_migration.create_vector_store')
    def test_migrate_success(self, mock_create_store, source_store_with_data, empty_target_store):
        """测试成功迁移"""
        migrator = VectorStoreMigrator()
        
        # 模拟工厂创建存储
        mock_create_store.side_effect = [source_store_with_data, empty_target_store]
        
        source_config = {"type": "memory"}
        target_config = {"type": "faiss"}
        
        result = migrator.migrate(
            source_config,
            target_config,
            batch_size=10,
            verify_migration=True,
            backup_before_migration=False
        )
        
        assert result == True
        assert empty_target_store.get_document_count() == 3
        
        # 验证文档内容
        migrated_doc = empty_target_store.get_document_by_id("doc1")
        assert migrated_doc is not None
        assert migrated_doc.content == "Document about AI"
    
    @patch('utils.vector_migration.create_vector_store')
    def test_migrate_with_backup(self, mock_create_store, source_store_with_data, empty_target_store, temp_backup_dir):
        """测试带备份的迁移"""
        migrator = VectorStoreMigrator()
        
        mock_create_store.side_effect = [source_store_with_data, empty_target_store]
        
        # 模拟备份方法
        with patch.object(migrator, 'backup_storage') as mock_backup:
            mock_backup.return_value = os.path.join(temp_backup_dir, "backup")
            
            result = migrator.migrate(
                {"type": "memory"},
                {"type": "faiss"},
                backup_before_migration=True
            )
        
        assert result == True
        assert mock_backup.called
    
    @patch('utils.vector_migration.create_vector_store')
    def test_migrate_backup_failure(self, mock_create_store, source_store_with_data):
        """测试备份失败时的迁移"""
        migrator = VectorStoreMigrator()
        
        mock_create_store.return_value = source_store_with_data
        
        with patch.object(migrator, 'backup_storage') as mock_backup:
            mock_backup.return_value = None  # 备份失败
            
            result = migrator.migrate(
                {"type": "memory"},
                {"type": "faiss"},
                backup_before_migration=True
            )
        
        assert result == False
    
    @patch('utils.vector_migration.create_vector_store')
    def test_migrate_verification_failure(self, mock_create_store, source_store_with_data, empty_target_store):
        """测试验证失败时的迁移"""
        migrator = VectorStoreMigrator()
        
        mock_create_store.side_effect = [source_store_with_data, empty_target_store]
        
        with patch.object(migrator, 'verify_migration') as mock_verify:
            mock_verify.return_value = False  # 验证失败
            
            result = migrator.migrate(
                {"type": "memory"},
                {"type": "faiss"},
                verify_migration=True
            )
        
        assert result == False
    
    def test_execute_migration_empty_source(self):
        """测试空源存储的迁移"""
        migrator = VectorStoreMigrator()
        
        empty_source = MockVectorStore({"type": "mock"})
        empty_source.initialize()
        
        empty_target = MockVectorStore({"type": "mock"})
        empty_target.initialize()
        
        result = migrator._execute_migration(empty_source, empty_target, 10)
        
        assert result == True
    
    def test_execute_migration_with_data(self, source_store_with_data, empty_target_store):
        """测试有数据的迁移执行"""
        migrator = VectorStoreMigrator()
        
        result = migrator._execute_migration(source_store_with_data, empty_target_store, 2)
        
        assert result == True
        assert empty_target_store.get_document_count() == 3
    
    def test_backup_storage_success(self, source_store_with_data, temp_backup_dir):
        """测试成功备份存储"""
        migrator = VectorStoreMigrator()
        
        backup_name = "test_backup"
        backup_path = migrator.backup_storage(source_store_with_data, backup_name, temp_backup_dir)
        
        assert backup_path is not None
        assert os.path.exists(backup_path)
        
        # 验证备份文件
        assert os.path.exists(os.path.join(backup_path, "backup_metadata.json"))
        assert os.path.exists(os.path.join(backup_path, "backup.checksum"))
        assert os.path.exists(os.path.join(backup_path, "mock_index.bin"))
        
        # 验证元数据
        metadata_path = os.path.join(backup_path, "backup_metadata.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        assert metadata["document_count"] == 3
        assert metadata["backup_version"] == "1.0"
        assert "backup_time" in metadata
    
    def test_backup_storage_save_failure(self, temp_backup_dir):
        """测试保存失败时的备份"""
        migrator = VectorStoreMigrator()
        
        # 创建一个保存总是失败的存储
        failing_store = MockVectorStore({"type": "mock"})
        failing_store.initialize()
        failing_store.save_index = lambda path: False  # 保存失败
        
        backup_path = migrator.backup_storage(failing_store, "test", temp_backup_dir)
        
        assert backup_path is None
    
    def test_restore_storage_success(self, empty_target_store, temp_backup_dir):
        """测试成功恢复存储"""
        migrator = VectorStoreMigrator()
        
        # 创建模拟备份
        backup_path = os.path.join(temp_backup_dir, "test_backup")
        os.makedirs(backup_path)
        
        # 创建备份元数据
        metadata = {
            "backup_time": datetime.now().isoformat(),
            "document_count": 2,
            "backup_version": "1.0"
        }
        
        metadata_path = os.path.join(backup_path, "backup_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
        
        # 创建索引文件
        index_path = os.path.join(backup_path, "mock_index.bin")
        with open(index_path, 'w') as f:
            f.write("mock data")
        
        # 创建校验和
        checksum = migrator._calculate_backup_checksum(backup_path)
        checksum_path = os.path.join(backup_path, "backup.checksum")
        with open(checksum_path, 'w') as f:
            f.write(checksum)
        
        # 使用模拟工厂创建目标存储
        with patch('utils.vector_migration.create_vector_store') as mock_create:
            mock_create.return_value = empty_target_store
            
            result = migrator.restore_storage(
                backup_path,
                {"type": "mock"},
                verify_backup=True
            )
        
        assert result == True
        assert len(empty_target_store.load_calls) == 1
    
    def test_restore_storage_missing_files(self, temp_backup_dir):
        """测试备份文件缺失时的恢复"""
        migrator = VectorStoreMigrator()
        
        # 创建不完整的备份目录
        backup_path = os.path.join(temp_backup_dir, "incomplete_backup")
        os.makedirs(backup_path)
        
        result = migrator.restore_storage(
            backup_path,
            {"type": "mock"},
            verify_backup=True
        )
        
        assert result == False
    
    def test_verify_migration_count_mismatch(self, source_store_with_data):
        """测试文档数量不匹配的验证"""
        migrator = VectorStoreMigrator()
        
        target_store = MockVectorStore({"type": "mock"})
        target_store.initialize()
        # 目标存储为空，数量不匹配
        
        result = migrator.verify_migration(source_store_with_data, target_store)
        
        assert result == False
    
    def test_verify_migration_empty_stores(self):
        """测试空存储的验证"""
        migrator = VectorStoreMigrator()
        
        empty_source = MockVectorStore({"type": "mock"})
        empty_source.initialize()
        
        empty_target = MockVectorStore({"type": "mock"})
        empty_target.initialize()
        
        result = migrator.verify_migration(empty_source, empty_target)
        
        assert result == True
    
    def test_verify_backup_success(self, temp_backup_dir):
        """测试成功的备份验证"""
        migrator = VectorStoreMigrator()
        
        # 创建有效的备份
        backup_path = os.path.join(temp_backup_dir, "valid_backup")
        os.makedirs(backup_path)
        
        # 创建必要文件
        metadata_path = os.path.join(backup_path, "backup_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({"test": "data"}, f)
        
        test_file = os.path.join(backup_path, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test data")
        
        # 计算和保存校验和
        checksum = migrator._calculate_backup_checksum(backup_path)
        checksum_path = os.path.join(backup_path, "backup.checksum")
        with open(checksum_path, 'w') as f:
            f.write(checksum)
        
        result = migrator.verify_backup(backup_path)
        
        assert result == True
    
    def test_verify_backup_missing_files(self, temp_backup_dir):
        """测试缺失文件的备份验证"""
        migrator = VectorStoreMigrator()
        
        backup_path = os.path.join(temp_backup_dir, "incomplete")
        os.makedirs(backup_path)
        
        result = migrator.verify_backup(backup_path)
        
        assert result == False
    
    def test_verify_backup_checksum_mismatch(self, temp_backup_dir):
        """测试校验和不匹配的备份验证"""
        migrator = VectorStoreMigrator()
        
        backup_path = os.path.join(temp_backup_dir, "corrupted")
        os.makedirs(backup_path)
        
        # 创建必要文件
        metadata_path = os.path.join(backup_path, "backup_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({"test": "data"}, f)
        
        # 保存错误的校验和
        checksum_path = os.path.join(backup_path, "backup.checksum")
        with open(checksum_path, 'w') as f:
            f.write("wrong_checksum")
        
        result = migrator.verify_backup(backup_path)
        
        assert result == False
    
    def test_calculate_backup_checksum(self, temp_backup_dir):
        """测试备份校验和计算"""
        migrator = VectorStoreMigrator()
        
        backup_path = os.path.join(temp_backup_dir, "test_checksum")
        os.makedirs(backup_path)
        
        # 创建测试文件
        test_files = ["file1.txt", "file2.txt", "backup.checksum"]  # 校验和文件应被忽略
        file_contents = ["content1", "content2", "should_be_ignored"]
        
        for filename, content in zip(test_files, file_contents):
            with open(os.path.join(backup_path, filename), 'w') as f:
                f.write(content)
        
        checksum1 = migrator._calculate_backup_checksum(backup_path)
        checksum2 = migrator._calculate_backup_checksum(backup_path)
        
        # 相同内容应该产生相同校验和
        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA256 hex digest length
    
    def test_list_backups_empty_directory(self, temp_backup_dir):
        """测试空备份目录的列表"""
        migrator = VectorStoreMigrator()
        
        backups = migrator.list_backups(temp_backup_dir)
        
        assert backups == []
    
    def test_list_backups_with_valid_backups(self, temp_backup_dir):
        """测试包含有效备份的目录列表"""
        migrator = VectorStoreMigrator()
        
        # 创建两个备份
        backup_names = ["backup1", "backup2"]
        for backup_name in backup_names:
            backup_path = os.path.join(temp_backup_dir, backup_name)
            os.makedirs(backup_path)
            
            metadata = {
                "backup_time": f"2024-01-{backup_name[-1]:02d}T10:00:00",
                "document_count": 10,
                "storage_info": {"storage_type": "TestStore"}
            }
            
            metadata_path = os.path.join(backup_path, "backup_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f)
        
        backups = migrator.list_backups(temp_backup_dir)
        
        assert len(backups) == 2
        
        # 验证备份信息结构
        for backup in backups:
            assert "name" in backup
            assert "path" in backup
            assert "backup_time" in backup
            assert "document_count" in backup
            assert "storage_type" in backup
            assert "size_mb" in backup
    
    def test_cleanup_old_backups(self, temp_backup_dir):
        """测试清理旧备份"""
        migrator = VectorStoreMigrator()
        
        # 创建5个备份
        backup_names = [f"backup_{i}" for i in range(5)]
        for backup_name in backup_names:
            backup_path = os.path.join(temp_backup_dir, backup_name)
            os.makedirs(backup_path)
            
            metadata = {
                "backup_time": f"2024-01-{len(backup_names)-int(backup_name.split('_')[1]):02d}T10:00:00",
                "document_count": 10
            }
            
            metadata_path = os.path.join(backup_path, "backup_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        
        # 保留3个，删除2个
        deleted_count = migrator.cleanup_old_backups(temp_backup_dir, keep_count=3)
        
        assert deleted_count == 2
        
        # 验证剩余备份数量
        remaining_backups = migrator.list_backups(temp_backup_dir)
        assert len(remaining_backups) == 3
    
    def test_cleanup_old_backups_under_limit(self, temp_backup_dir):
        """测试备份数量未超过限制时的清理"""
        migrator = VectorStoreMigrator()
        
        # 创建2个备份，限制为5个
        for i in range(2):
            backup_path = os.path.join(temp_backup_dir, f"backup_{i}")
            os.makedirs(backup_path)
            
            metadata = {"backup_time": f"2024-01-0{i+1}T10:00:00", "document_count": 10}
            metadata_path = os.path.join(backup_path, "backup_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        
        deleted_count = migrator.cleanup_old_backups(temp_backup_dir, keep_count=5)
        
        assert deleted_count == 0


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    @patch('utils.vector_migration.VectorStoreMigrator.migrate')
    def test_migrate_vector_storage_function(self, mock_migrate):
        """测试migrate_vector_storage便捷函数"""
        mock_migrate.return_value = True
        
        source_config = {"type": "memory"}
        target_config = {"type": "faiss"}
        
        result = migrate_vector_storage(source_config, target_config, batch_size=500)
        
        assert result == True
        assert mock_migrate.called
        
        # 验证调用参数
        call_args = mock_migrate.call_args
        assert call_args[0][0] == source_config
        assert call_args[0][1] == target_config
        assert call_args[0][2] == 500
    
    @patch('utils.vector_migration.VectorStoreMigrator.backup_storage')
    def test_backup_vector_storage_function(self, mock_backup):
        """测试backup_vector_storage便捷函数"""
        mock_backup.return_value = "/path/to/backup"
        
        mock_store = Mock()
        backup_name = "test_backup"
        
        result = backup_vector_storage(mock_store, backup_name)
        
        assert result == "/path/to/backup"
        assert mock_backup.called
        
        call_args = mock_backup.call_args
        assert call_args[0][0] == mock_store
        assert call_args[0][1] == backup_name


class TestEdgeCases:
    """边界情况测试"""
    
    def test_migrate_with_exception(self):
        """测试迁移过程中发生异常"""
        migrator = VectorStoreMigrator()
        
        with patch('utils.vector_migration.create_vector_store') as mock_create:
            mock_create.side_effect = Exception("Connection failed")
            
            result = migrator.migrate(
                {"type": "memory"},
                {"type": "faiss"}
            )
        
        assert result == False
    
    def test_backup_with_io_error(self, temp_backup_dir):
        """测试备份过程中的IO错误"""
        migrator = VectorStoreMigrator()
        
        store = MockVectorStore({"type": "mock"})
        store.initialize()
        
        # 模拟写入权限问题
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            backup_path = migrator.backup_storage(store, "test", temp_backup_dir)
        
        assert backup_path is None
    
    def test_get_directory_size_nonexistent(self):
        """测试不存在目录的大小计算"""
        migrator = VectorStoreMigrator()
        
        size = migrator._get_directory_size("/nonexistent/path")
        
        assert size == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])