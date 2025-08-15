"""
配置系统集成测试

测试配置验证、版本管理、迁移工具和环境管理的集成功能。
验证配置系统的端到端工作流程。
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from core.config_runtime_validator import (
    RuntimeConfigValidator, ValidationLevel, ValidationSeverity,
    RangeConstraint, PatternConstraint, DependencyConstraint
)
from core.config_version_manager import ConfigVersionManager, ChangeType
from core.config_migration_tool import ConfigMigrationTool, EnvironmentType as MigrationEnvType
from core.config_environment_manager import ConfigEnvironmentManager, EnvironmentType

class TestConfigSystemIntegration:
    """配置系统集成测试"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """临时配置目录"""
        temp_dir = tempfile.mkdtemp(prefix='config_test_')
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_config(self):
        """示例配置"""
        return {
            "version": "2.0.0",
            "storage": {
                "base_path": "./data",
                "output_path": "./output"
            },
            "processors": {
                "pre_processor": {
                    "enabled": True,
                    "config": {"batch_size": 10}
                },
                "ocr_processor": {
                    "enabled": True,
                    "config": {"batch_size": 5}
                }
            },
            "llm": {
                "type": "openai",
                "model": "gpt-3.5-turbo",
                "settings": {
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "api_key": "test-key"
                }
            },
            "vector_db": {
                "document_store": {
                    "type": "memory",
                    "embedding_dim": 128,
                    "similarity": "cosine"
                }
            }
        }
    
    @pytest.fixture
    def old_version_config(self):
        """旧版本配置（需要迁移）"""
        return {
            "storage": {
                "data_path": "./old_data",
                "result_path": "./old_results"
            },
            "processors": {
                "pre_processor": True,
                "ocr_processor": False
            },
            "generator": {
                "model_name": "gpt-3.5-turbo",
                "params": {
                    "temperature": 1.5,
                    "max_tokens": 500
                }
            },
            "document_store": {
                "type": "faiss",
                "embedding_dim": 1536
            }
        }
    
    def test_runtime_validation_integration(self, temp_config_dir, sample_config):
        """测试运行时验证集成"""
        # 创建验证器
        validator = RuntimeConfigValidator(ValidationLevel.ENTERPRISE)
        
        # 验证有效配置
        is_valid, results = validator.validate_config(sample_config)
        assert is_valid
        
        # 验证无效配置
        invalid_config = sample_config.copy()
        invalid_config["llm"]["settings"]["temperature"] = 5.0  # 超出范围
        invalid_config["storage"]["base_path"] = ""  # 空路径
        
        is_valid, results = validator.validate_config(invalid_config)
        assert not is_valid
        
        # 检查错误类型
        error_results = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(error_results) > 0
        
        # 生成验证报告
        report = validator.generate_validation_report(results)
        assert "ERROR" in report or "WARNING" in report
    
    def test_version_management_integration(self, temp_config_dir, sample_config):
        """测试版本管理集成"""
        config_file = Path(temp_config_dir) / "config.json"
        
        # 创建版本管理器
        version_manager = ConfigVersionManager(str(config_file))
        
        # 保存配置并创建版本
        version_id, changes = version_manager.save_config_with_version(
            sample_config, 
            description="初始配置",
            user="test_user",
            tags=["initial"]
        )
        
        assert version_id
        assert config_file.exists()
        
        # 修改配置
        modified_config = sample_config.copy()
        modified_config["llm"]["settings"]["temperature"] = 0.5
        modified_config["storage"]["base_path"] = "./new_data"
        
        # 保存修改并创建新版本
        new_version_id, new_changes = version_manager.save_config_with_version(
            modified_config,
            description="更新配置",
            user="test_user",
            tags=["update"]
        )
        
        assert new_version_id != version_id
        assert len(new_changes) > 0
        
        # 验证变更记录
        update_changes = [c for c in new_changes if c.change_type == ChangeType.UPDATED]
        assert len(update_changes) >= 2  # temperature和base_path
        
        # 测试版本恢复
        restore_success = version_manager.restore_version(version_id, "test_user", "测试恢复")
        assert restore_success
        
        # 验证恢复后的配置
        with open(config_file, 'r') as f:
            restored_config = json.load(f)
        
        assert restored_config["llm"]["settings"]["temperature"] == 0.7
        assert restored_config["storage"]["base_path"] == "./data"
        
        # 测试版本比较
        comparison = version_manager.compare_versions(version_id, new_version_id)
        assert "changes" in comparison
        assert comparison["summary"]["total_changes"] > 0
    
    def test_migration_tool_integration(self, temp_config_dir, old_version_config):
        """测试迁移工具集成"""
        config_file = Path(temp_config_dir) / "config.json"
        
        # 保存旧版本配置
        with open(config_file, 'w') as f:
            json.dump(old_version_config, f, indent=2)
        
        # 创建迁移工具
        migration_tool = ConfigMigrationTool()
        
        # 检测版本
        detected_version = migration_tool.detect_config_version(old_version_config)
        assert detected_version == "1.0.0"
        
        # 执行迁移
        result = migration_tool.migrate_config_file(str(config_file), "2.0.0", backup=True)
        
        assert result.status.value == "completed"
        assert result.from_version == "1.0.0"
        assert result.to_version == "2.0.0"
        assert len(result.migrated_paths) > 0
        assert result.backup_path  # 确保创建了备份
        
        # 验证迁移后的配置
        with open(config_file, 'r') as f:
            migrated_config = json.load(f)
        
        # 检查路径重命名
        assert "base_path" in migrated_config["storage"]
        assert "output_path" in migrated_config["storage"]
        assert "data_path" not in migrated_config["storage"]
        
        # 检查处理器结构化
        assert isinstance(migrated_config["processors"]["pre_processor"], dict)
        assert migrated_config["processors"]["pre_processor"]["enabled"] is True
        
        # 检查LLM配置重命名
        assert "llm" in migrated_config
        assert "generator" not in migrated_config
        
        # 检查向量数据库重构
        assert "vector_db" in migrated_config
        assert "document_store" not in migrated_config
        
        # 验证迁移结果
        validation_issues = migration_tool.validate_migration(old_version_config, migrated_config)
        # 可能有一些预期的差异，但不应该有严重问题
        assert len(validation_issues) < 5
    
    def test_environment_management_integration(self, temp_config_dir, sample_config):
        """测试环境管理集成"""
        # 创建环境管理器
        env_manager = ConfigEnvironmentManager(temp_config_dir)
        
        # 验证默认环境创建
        environments = env_manager.list_environments()
        assert len(environments) >= 3  # development, testing, production
        
        env_names = [env['name'] for env in environments]
        assert 'development' in env_names
        assert 'testing' in env_names
        assert 'production' in env_names
        
        # 测试环境切换
        switch_success = env_manager.set_environment('development')
        assert switch_success
        assert env_manager.current_environment == 'development'
        
        # 获取环境配置
        dev_config = env_manager.get_environment_config('development')
        assert dev_config
        assert dev_config['environment']['name'] == 'development'
        assert dev_config['environment']['debug'] is True
        
        # 创建自定义环境
        custom_env_created = env_manager.create_environment(
            'custom_test',
            EnvironmentType.CUSTOM,
            '自定义测试环境',
            sample_config
        )
        assert custom_env_created
        
        # 验证自定义环境
        custom_config = env_manager.get_environment_config('custom_test')
        assert custom_config
        assert custom_config['llm']['model'] == 'gpt-3.5-turbo'
        
        # 测试环境复制
        copy_success = env_manager.copy_environment('custom_test', 'custom_copy')
        assert copy_success
        
        copied_config = env_manager.get_environment_config('custom_copy')
        assert copied_config
        assert copied_config['llm']['model'] == 'gpt-3.5-turbo'
        
        # 测试配置更新
        config_updates = {
            'llm': {
                'model': 'gpt-4',
                'settings': {
                    'temperature': 0.3
                }
            }
        }
        
        update_success = env_manager.update_environment_config('custom_test', config_updates)
        assert update_success
        
        updated_config = env_manager.get_environment_config('custom_test')
        assert updated_config['llm']['model'] == 'gpt-4'
        assert updated_config['llm']['settings']['temperature'] == 0.3
        
        # 测试环境验证
        is_valid, errors = env_manager.validate_environment('custom_test')
        assert is_valid
        assert len(errors) == 0
        
        # 测试环境差异比较
        diff = env_manager.get_environment_diff('development', 'custom_test')
        assert diff
        assert 'summary' in diff
        assert diff['summary']['total_differences'] > 0
    
    def test_full_config_workflow_integration(self, temp_config_dir):
        """测试完整配置工作流程集成"""
        config_file = Path(temp_config_dir) / "config.json"
        
        # 步骤1: 创建环境管理器并设置开发环境
        env_manager = ConfigEnvironmentManager(temp_config_dir)
        env_manager.set_environment('development')
        dev_config = env_manager.get_environment_config('development')
        
        # 步骤2: 使用版本管理器管理配置
        version_manager = ConfigVersionManager(str(config_file))
        version_id = version_manager.create_version(
            dev_config,
            "开发环境初始配置",
            "integration_test"
        )
        
        # 步骤3: 验证配置
        validator = RuntimeConfigValidator(ValidationLevel.STANDARD)
        is_valid, results = validator.validate_config(dev_config)
        assert is_valid or len([r for r in results if r.severity == ValidationSeverity.ERROR]) == 0
        
        # 步骤4: 修改配置
        modified_config = dev_config.copy()
        modified_config['llm']['model'] = 'gpt-4'
        modified_config['processors']['new_processor'] = {
            'enabled': True,
            'config': {'batch_size': 20}
        }
        
        # 步骤5: 保存新版本
        new_version_id, changes = version_manager.save_config_with_version(
            modified_config,
            "添加新处理器",
            "integration_test"
        )
        
        assert len(changes) >= 2  # 修改model和添加new_processor
        
        # 步骤6: 验证修改后的配置
        is_valid, results = validator.validate_config(modified_config)
        # 可能有警告但不应有错误
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0
        
        # 步骤7: 创建生产环境配置
        prod_config = modified_config.copy()
        prod_config['environment'] = {
            'name': 'production',
            'debug': False
        }
        prod_config['logging']['level'] = 'WARNING'
        
        env_manager.create_environment(
            'production_custom',
            EnvironmentType.PRODUCTION,
            '自定义生产环境',
            prod_config
        )
        
        # 步骤8: 验证环境差异
        diff = env_manager.get_environment_diff('development', 'production_custom')
        assert diff['summary']['total_differences'] > 0
        
        # 步骤9: 导出和导入测试
        export_path = Path(temp_config_dir) / "exported_env.json"
        export_success = env_manager.export_environment('production_custom', str(export_path))
        assert export_success
        assert export_path.exists()
        
        imported_env = env_manager.import_environment(str(export_path), 'imported_prod')
        assert imported_env == 'imported_prod'
        
        # 步骤10: 验证导入的环境
        imported_config = env_manager.get_environment_config('imported_prod')
        assert imported_config
        assert imported_config['llm']['model'] == 'gpt-4'
        
        # 步骤11: 版本统计和清理
        stats = version_manager.get_statistics()
        assert stats['total_versions'] >= 2
        assert stats['latest_version']
        
        history = version_manager.get_change_history(limit=10)
        assert len(history) > 0
    
    def test_error_handling_and_recovery(self, temp_config_dir):
        """测试错误处理和恢复机制"""
        config_file = Path(temp_config_dir) / "config.json"
        
        # 测试无效配置处理
        invalid_config = {
            "storage": {
                "base_path": None  # 无效路径
            },
            "processors": "invalid"  # 应该是字典
        }
        
        validator = RuntimeConfigValidator(ValidationLevel.STRICT)
        is_valid, results = validator.validate_config(invalid_config)
        assert not is_valid
        
        error_count = len([r for r in results if r.severity == ValidationSeverity.ERROR])
        assert error_count > 0
        
        # 测试版本管理器的错误恢复
        version_manager = ConfigVersionManager(str(config_file))
        
        # 尝试恢复不存在的版本
        restore_result = version_manager.restore_version("nonexistent_version")
        assert not restore_result
        
        # 测试环境管理器的错误处理
        env_manager = ConfigEnvironmentManager(temp_config_dir)
        
        # 尝试切换到不存在的环境
        switch_result = env_manager.set_environment("nonexistent_env")
        assert not switch_result
        
        # 尝试删除当前环境
        env_manager.set_environment('development')
        delete_result = env_manager.delete_environment('development')
        assert not delete_result  # 不能删除当前环境
        
        # 测试迁移工具的错误处理
        migration_tool = ConfigMigrationTool()
        
        # 尝试迁移无效配置
        result = migration_tool.migrate_config(invalid_config, "2.0.0")
        # 应该能处理但可能有警告
        assert result.status.value in ["completed", "failed"]
    
    @pytest.mark.asyncio
    async def test_concurrent_access_handling(self, temp_config_dir, sample_config):
        """测试并发访问处理"""
        import asyncio
        config_file = Path(temp_config_dir) / "config.json"
        
        # 并发版本创建测试
        version_manager = ConfigVersionManager(str(config_file))
        
        async def create_version_async(config, description, user):
            return version_manager.create_version(config, description, user)
        
        # 创建多个并发版本
        tasks = []
        for i in range(5):
            modified_config = sample_config.copy()
            modified_config['test_field'] = f'value_{i}'
            
            task = create_version_async(
                modified_config,
                f"并发测试版本 {i}",
                f"user_{i}"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 验证所有版本都创建成功
        successful_results = [r for r in results if isinstance(r, str)]
        assert len(successful_results) == 5
        
        # 验证版本唯一性
        assert len(set(successful_results)) == 5