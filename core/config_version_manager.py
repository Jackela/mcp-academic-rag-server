"""
配置版本管理器

提供配置版本控制、变更历史记录、回滚功能和配置审计。
支持自动备份、变更追踪和配置比较功能。
"""

import os
import json
import time
import shutil
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ChangeType(Enum):
    """配置变更类型"""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    RESTORED = "restored"

@dataclass
class ConfigChange:
    """配置变更记录"""
    timestamp: str
    version: str
    change_type: ChangeType
    path: str
    old_value: Any
    new_value: Any
    user: Optional[str] = None
    reason: Optional[str] = None
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'version': self.version,
            'change_type': self.change_type.value,
            'path': self.path,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'user': self.user,
            'reason': self.reason,
            'checksum': self.checksum
        }

@dataclass
class ConfigVersion:
    """配置版本信息"""
    version: str
    timestamp: str
    config_data: Dict[str, Any]
    checksum: str
    size: int
    user: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'timestamp': self.timestamp,
            'config_data': self.config_data,
            'checksum': self.checksum,
            'size': self.size,
            'user': self.user,
            'description': self.description,
            'tags': self.tags or []
        }

class ConfigVersionManager:
    """配置版本管理器"""
    
    def __init__(self, config_path: str, versions_dir: str = None, max_versions: int = 100):
        self.config_path = Path(config_path)
        self.versions_dir = Path(versions_dir) if versions_dir else self.config_path.parent / 'versions'
        self.max_versions = max_versions
        
        # 创建版本目录
        self.versions_dir.mkdir(exist_ok=True)
        
        # 历史文件路径
        self.history_file = self.versions_dir / 'change_history.json'
        self.metadata_file = self.versions_dir / 'versions_metadata.json'
        
        # 初始化历史记录
        self._ensure_history_files()
    
    def _ensure_history_files(self):
        """确保历史文件存在"""
        if not self.history_file.exists():
            self._save_json(self.history_file, [])
        
        if not self.metadata_file.exists():
            self._save_json(self.metadata_file, {})
    
    def _save_json(self, file_path: Path, data: Any):
        """安全保存JSON文件"""
        try:
            temp_path = file_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            # 原子性替换
            temp_path.replace(file_path)
            
        except Exception as e:
            logger.error(f"保存JSON文件失败 {file_path}: {e}")
            raise
    
    def _load_json(self, file_path: Path, default: Any = None) -> Any:
        """安全加载JSON文件"""
        try:
            if not file_path.exists():
                return default
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        except Exception as e:
            logger.error(f"加载JSON文件失败 {file_path}: {e}")
            return default
    
    def _calculate_checksum(self, config_data: Dict[str, Any]) -> str:
        """计算配置校验和"""
        config_str = json.dumps(config_data, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()
    
    def _generate_version_id(self) -> str:
        """生成版本ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"v{timestamp}_{random_suffix}"
    
    def _detect_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> List[ConfigChange]:
        """检测配置变更"""
        changes = []
        version = self._generate_version_id()
        timestamp = datetime.now().isoformat()
        
        def compare_recursive(old_dict: Dict[str, Any], new_dict: Dict[str, Any], path: str = ""):
            # 检查新增和修改
            for key, new_value in new_dict.items():
                current_path = f"{path}.{key}" if path else key
                
                if key not in old_dict:
                    # 新增
                    changes.append(ConfigChange(
                        timestamp=timestamp,
                        version=version,
                        change_type=ChangeType.CREATED,
                        path=current_path,
                        old_value=None,
                        new_value=new_value
                    ))
                elif old_dict[key] != new_value:
                    if isinstance(old_dict[key], dict) and isinstance(new_value, dict):
                        # 递归比较嵌套字典
                        compare_recursive(old_dict[key], new_value, current_path)
                    else:
                        # 修改
                        changes.append(ConfigChange(
                            timestamp=timestamp,
                            version=version,
                            change_type=ChangeType.UPDATED,
                            path=current_path,
                            old_value=old_dict[key],
                            new_value=new_value
                        ))
            
            # 检查删除
            for key, old_value in old_dict.items():
                if key not in new_dict:
                    current_path = f"{path}.{key}" if path else key
                    changes.append(ConfigChange(
                        timestamp=timestamp,
                        version=version,
                        change_type=ChangeType.DELETED,
                        path=current_path,
                        old_value=old_value,
                        new_value=None
                    ))
        
        compare_recursive(old_config, new_config)
        return changes
    
    def create_version(self, config_data: Dict[str, Any], description: str = None, 
                      user: str = None, tags: List[str] = None) -> str:
        """创建新版本"""
        try:
            version_id = self._generate_version_id()
            timestamp = datetime.now().isoformat()
            checksum = self._calculate_checksum(config_data)
            config_size = len(json.dumps(config_data, default=str))
            
            # 创建版本对象
            version = ConfigVersion(
                version=version_id,
                timestamp=timestamp,
                config_data=config_data,
                checksum=checksum,
                size=config_size,
                user=user,
                description=description,
                tags=tags
            )
            
            # 保存版本文件
            version_file = self.versions_dir / f"{version_id}.json"
            self._save_json(version_file, version.to_dict())
            
            # 更新元数据
            metadata = self._load_json(self.metadata_file, {})
            metadata[version_id] = {
                'timestamp': timestamp,
                'checksum': checksum,
                'size': config_size,
                'user': user,
                'description': description,
                'tags': tags or []
            }
            self._save_json(self.metadata_file, metadata)
            
            # 清理旧版本
            self._cleanup_old_versions()
            
            logger.info(f"创建配置版本: {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"创建版本失败: {e}")
            raise
    
    def save_config_with_version(self, config_data: Dict[str, Any], description: str = None,
                                user: str = None, tags: List[str] = None) -> Tuple[str, List[ConfigChange]]:
        """保存配置并创建版本"""
        try:
            # 加载当前配置
            old_config = {}
            if self.config_path.exists():
                old_config = self._load_json(self.config_path, {})
            
            # 检测变更
            changes = self._detect_changes(old_config, config_data)
            
            # 保存新配置
            self._save_json(self.config_path, config_data)
            
            # 创建版本
            version_id = self.create_version(config_data, description, user, tags)
            
            # 记录变更历史
            if changes:
                self._record_changes(changes)
            
            logger.info(f"保存配置并创建版本: {version_id}, 变更数量: {len(changes)}")
            return version_id, changes
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise
    
    def _record_changes(self, changes: List[ConfigChange]):
        """记录变更历史"""
        try:
            history = self._load_json(self.history_file, [])
            
            for change in changes:
                history.append(change.to_dict())
            
            self._save_json(self.history_file, history)
            
        except Exception as e:
            logger.error(f"记录变更历史失败: {e}")
    
    def get_version(self, version_id: str) -> Optional[ConfigVersion]:
        """获取指定版本"""
        try:
            version_file = self.versions_dir / f"{version_id}.json"
            if not version_file.exists():
                return None
            
            version_data = self._load_json(version_file)
            if not version_data:
                return None
            
            return ConfigVersion(**version_data)
            
        except Exception as e:
            logger.error(f"获取版本失败 {version_id}: {e}")
            return None
    
    def list_versions(self, limit: int = None, tags: List[str] = None) -> List[Dict[str, Any]]:
        """列出版本"""
        try:
            metadata = self._load_json(self.metadata_file, {})
            
            versions = []
            for version_id, info in metadata.items():
                # 标签过滤
                if tags and not any(tag in info.get('tags', []) for tag in tags):
                    continue
                
                versions.append({
                    'version': version_id,
                    'timestamp': info['timestamp'],
                    'checksum': info['checksum'],
                    'size': info['size'],
                    'user': info.get('user'),
                    'description': info.get('description'),
                    'tags': info.get('tags', [])
                })
            
            # 按时间排序
            versions.sort(key=lambda x: x['timestamp'], reverse=True)
            
            if limit:
                versions = versions[:limit]
            
            return versions
            
        except Exception as e:
            logger.error(f"列出版本失败: {e}")
            return []
    
    def restore_version(self, version_id: str, user: str = None, reason: str = None) -> bool:
        """恢复到指定版本"""
        try:
            version = self.get_version(version_id)
            if not version:
                logger.error(f"版本不存在: {version_id}")
                return False
            
            # 备份当前配置
            if self.config_path.exists():
                backup_description = f"恢复到 {version_id} 前的备份"
                current_config = self._load_json(self.config_path, {})
                self.create_version(current_config, backup_description, user, ['backup'])
            
            # 恢复配置
            self._save_json(self.config_path, version.config_data)
            
            # 记录恢复操作
            restore_change = ConfigChange(
                timestamp=datetime.now().isoformat(),
                version=self._generate_version_id(),
                change_type=ChangeType.RESTORED,
                path="root",
                old_value=None,
                new_value=version_id,
                user=user,
                reason=reason or f"恢复到版本 {version_id}"
            )
            
            self._record_changes([restore_change])
            
            logger.info(f"成功恢复到版本: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"恢复版本失败 {version_id}: {e}")
            return False
    
    def get_change_history(self, limit: int = None, path_filter: str = None, 
                          change_type_filter: ChangeType = None) -> List[Dict[str, Any]]:
        """获取变更历史"""
        try:
            history = self._load_json(self.history_file, [])
            
            # 过滤
            filtered_history = []
            for change in history:
                # 路径过滤
                if path_filter and not change['path'].startswith(path_filter):
                    continue
                
                # 变更类型过滤
                if change_type_filter and change['change_type'] != change_type_filter.value:
                    continue
                
                filtered_history.append(change)
            
            # 按时间排序
            filtered_history.sort(key=lambda x: x['timestamp'], reverse=True)
            
            if limit:
                filtered_history = filtered_history[:limit]
            
            return filtered_history
            
        except Exception as e:
            logger.error(f"获取变更历史失败: {e}")
            return []
    
    def compare_versions(self, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """比较两个版本"""
        try:
            version1 = self.get_version(version1_id)
            version2 = self.get_version(version2_id)
            
            if not version1 or not version2:
                return {'error': '版本不存在'}
            
            changes = self._detect_changes(version1.config_data, version2.config_data)
            
            return {
                'version1': version1_id,
                'version2': version2_id,
                'changes': [change.to_dict() for change in changes],
                'summary': {
                    'total_changes': len(changes),
                    'created': len([c for c in changes if c.change_type == ChangeType.CREATED]),
                    'updated': len([c for c in changes if c.change_type == ChangeType.UPDATED]),
                    'deleted': len([c for c in changes if c.change_type == ChangeType.DELETED])
                }
            }
            
        except Exception as e:
            logger.error(f"比较版本失败 {version1_id} vs {version2_id}: {e}")
            return {'error': str(e)}
    
    def _cleanup_old_versions(self):
        """清理旧版本"""
        try:
            metadata = self._load_json(self.metadata_file, {})
            
            if len(metadata) <= self.max_versions:
                return
            
            # 按时间排序，保留最新的版本
            versions = sorted(metadata.items(), key=lambda x: x[1]['timestamp'], reverse=True)
            versions_to_keep = versions[:self.max_versions]
            versions_to_delete = versions[self.max_versions:]
            
            # 删除旧版本文件
            for version_id, _ in versions_to_delete:
                version_file = self.versions_dir / f"{version_id}.json"
                if version_file.exists():
                    version_file.unlink()
                
                # 从元数据中删除
                del metadata[version_id]
            
            # 保存更新后的元数据
            self._save_json(self.metadata_file, metadata)
            
            logger.info(f"清理了 {len(versions_to_delete)} 个旧版本")
            
        except Exception as e:
            logger.error(f"清理旧版本失败: {e}")
    
    def export_version(self, version_id: str, export_path: str) -> bool:
        """导出版本"""
        try:
            version = self.get_version(version_id)
            if not version:
                return False
            
            export_data = {
                'export_info': {
                    'exported_at': datetime.now().isoformat(),
                    'original_version': version_id,
                    'source_config': str(self.config_path)
                },
                'version_data': version.to_dict()
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"导出版本 {version_id} 到 {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出版本失败 {version_id}: {e}")
            return False
    
    def import_version(self, import_path: str, user: str = None) -> Optional[str]:
        """导入版本"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            version_data = import_data.get('version_data', {})
            if not version_data:
                logger.error("导入文件格式错误")
                return None
            
            # 创建新版本ID
            new_version_id = self._generate_version_id()
            version_data['version'] = new_version_id
            version_data['timestamp'] = datetime.now().isoformat()
            version_data['user'] = user
            
            # 保存版本
            version_file = self.versions_dir / f"{new_version_id}.json"
            self._save_json(version_file, version_data)
            
            # 更新元数据
            metadata = self._load_json(self.metadata_file, {})
            metadata[new_version_id] = {
                'timestamp': version_data['timestamp'],
                'checksum': version_data['checksum'],
                'size': version_data['size'],
                'user': user,
                'description': f"从 {import_path} 导入",
                'tags': ['imported']
            }
            self._save_json(self.metadata_file, metadata)
            
            logger.info(f"导入版本: {new_version_id}")
            return new_version_id
            
        except Exception as e:
            logger.error(f"导入版本失败: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            metadata = self._load_json(self.metadata_file, {})
            history = self._load_json(self.history_file, [])
            
            # 版本统计
            total_versions = len(metadata)
            total_size = sum(info['size'] for info in metadata.values())
            
            # 最新版本
            latest_version = None
            if metadata:
                latest_version = max(metadata.items(), key=lambda x: x[1]['timestamp'])
            
            # 变更统计
            change_stats = {}
            for change in history:
                change_type = change['change_type']
                change_stats[change_type] = change_stats.get(change_type, 0) + 1
            
            # 用户活动统计
            user_stats = {}
            for info in metadata.values():
                user = info.get('user', 'unknown')
                user_stats[user] = user_stats.get(user, 0) + 1
            
            return {
                'total_versions': total_versions,
                'total_size_bytes': total_size,
                'latest_version': latest_version[0] if latest_version else None,
                'latest_timestamp': latest_version[1]['timestamp'] if latest_version else None,
                'change_statistics': change_stats,
                'user_statistics': user_stats,
                'storage_path': str(self.versions_dir),
                'config_path': str(self.config_path)
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

# 便捷函数
def create_version_manager(config_path: str, versions_dir: str = None) -> ConfigVersionManager:
    """创建版本管理器实例"""
    return ConfigVersionManager(config_path, versions_dir)

def quick_backup(config_path: str, description: str = None) -> str:
    """快速备份当前配置"""
    manager = ConfigVersionManager(config_path)
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    return manager.create_version(
        config_data, 
        description or f"快速备份 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        tags=['quick-backup']
    )