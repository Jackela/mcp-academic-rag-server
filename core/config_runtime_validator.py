"""
运行时配置验证器

提供实时配置验证、配置约束检查和配置兼容性验证功能。
支持动态配置验证、依赖关系检查和性能影响评估。
"""

import logging
import os
import re
import time
from typing import Dict, Any, List, Optional, Tuple, Set, Union, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """验证级别"""
    BASIC = "basic"           # 基础语法验证
    STANDARD = "standard"     # 标准业务逻辑验证
    STRICT = "strict"         # 严格验证（包括性能影响）
    ENTERPRISE = "enterprise" # 企业级验证（包括安全和合规）

class ValidationSeverity(Enum):
    """验证严重程度"""
    INFO = "info"         # 信息性提示
    WARNING = "warning"   # 警告，不阻止运行
    ERROR = "error"       # 错误，阻止运行
    CRITICAL = "critical" # 严重错误，立即停止

@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    severity: ValidationSeverity
    path: str
    message: str
    suggestion: Optional[str] = None
    impact: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'severity': self.severity.value,
            'path': self.path,
            'message': self.message,
            'suggestion': self.suggestion,
            'impact': self.impact
        }

class ConfigConstraint:
    """配置约束基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def validate(self, value: Any, path: str, full_config: Dict[str, Any]) -> List[ValidationResult]:
        """验证配置值"""
        raise NotImplementedError

class RangeConstraint(ConfigConstraint):
    """数值范围约束"""
    
    def __init__(self, name: str, min_val: Union[int, float], max_val: Union[int, float], 
                 description: str = None):
        super().__init__(name, description or f"值必须在 {min_val} 到 {max_val} 之间")
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, value: Any, path: str, full_config: Dict[str, Any]) -> List[ValidationResult]:
        results = []
        
        if not isinstance(value, (int, float)):
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                path=path,
                message=f"值必须是数字类型，当前类型: {type(value).__name__}",
                suggestion="请使用数字值"
            ))
            return results
        
        if not (self.min_val <= value <= self.max_val):
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                path=path,
                message=f"值 {value} 超出范围 [{self.min_val}, {self.max_val}]",
                suggestion=f"请使用 {self.min_val} 到 {self.max_val} 之间的值"
            ))
        else:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                path=path,
                message=f"数值范围验证通过: {value}"
            ))
        
        return results

class PatternConstraint(ConfigConstraint):
    """正则表达式模式约束"""
    
    def __init__(self, name: str, pattern: str, description: str = None):
        super().__init__(name, description or f"必须匹配模式: {pattern}")
        self.pattern = re.compile(pattern)
        self.pattern_str = pattern
    
    def validate(self, value: Any, path: str, full_config: Dict[str, Any]) -> List[ValidationResult]:
        results = []
        
        if not isinstance(value, str):
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                path=path,
                message=f"值必须是字符串类型，当前类型: {type(value).__name__}",
                suggestion="请使用字符串值"
            ))
            return results
        
        if not self.pattern.match(value):
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                path=path,
                message=f"值 '{value}' 不匹配模式 '{self.pattern_str}'",
                suggestion="请检查格式并确保符合要求的模式"
            ))
        else:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                path=path,
                message=f"模式验证通过: {value}"
            ))
        
        return results

class DependencyConstraint(ConfigConstraint):
    """依赖关系约束"""
    
    def __init__(self, name: str, depends_on: List[str], description: str = None):
        super().__init__(name, description or f"依赖于: {', '.join(depends_on)}")
        self.depends_on = depends_on
    
    def validate(self, value: Any, path: str, full_config: Dict[str, Any]) -> List[ValidationResult]:
        results = []
        
        for dep_path in self.depends_on:
            if not self._get_nested_value(full_config, dep_path):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    path=path,
                    message=f"缺少依赖配置: {dep_path}",
                    suggestion=f"请先配置 {dep_path}",
                    impact="可能导致功能无法正常工作"
                ))
        
        if not results:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                path=path,
                message="依赖关系验证通过"
            ))
        
        return results
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """获取嵌套配置值"""
        keys = path.split('.')
        current = config
        
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        
        return current

class RuntimeConfigValidator:
    """运行时配置验证器"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.constraints: Dict[str, List[ConfigConstraint]] = {}
        self.custom_validators: Dict[str, Callable] = {}
        
        # 注册默认约束
        self._register_default_constraints()
    
    def _register_default_constraints(self):
        """注册默认配置约束"""
        # 存储路径约束
        self.add_constraint("storage.base_path", PatternConstraint(
            "storage_path", r'^[^<>:"|?*]+$', "存储路径不能包含非法字符"
        ))
        
        # 处理器配置约束
        self.add_constraint("processors.*.config.batch_size", RangeConstraint(
            "batch_size", 1, 1000, "批处理大小必须在1-1000之间"
        ))
        
        # LLM配置约束
        self.add_constraint("llm.settings.temperature", RangeConstraint(
            "temperature", 0.0, 2.0, "温度参数必须在0.0-2.0之间"
        ))
        
        self.add_constraint("llm.settings.max_tokens", RangeConstraint(
            "max_tokens", 1, 32000, "最大tokens必须在1-32000之间"
        ))
        
        # 向量数据库约束
        self.add_constraint("vector_db.document_store.embedding_dim", RangeConstraint(
            "embedding_dim", 64, 4096, "嵌入维度必须在64-4096之间"
        ))
        
        # 依赖关系约束
        self.add_constraint("processors.embedding_processor", DependencyConstraint(
            "embedding_dependency", ["llm.settings.api_key"], "嵌入处理器需要LLM API密钥"
        ))
    
    def add_constraint(self, path: str, constraint: ConfigConstraint):
        """添加配置约束"""
        if path not in self.constraints:
            self.constraints[path] = []
        self.constraints[path].append(constraint)
    
    def add_custom_validator(self, path: str, validator: Callable[[Any, str, Dict[str, Any]], List[ValidationResult]]):
        """添加自定义验证器"""
        self.custom_validators[path] = validator
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[ValidationResult]]:
        """验证完整配置"""
        all_results = []
        
        # 基础结构验证
        if self.validation_level.value in ['standard', 'strict', 'enterprise']:
            all_results.extend(self._validate_structure(config))
        
        # 约束验证
        all_results.extend(self._validate_constraints(config))
        
        # 自定义验证
        all_results.extend(self._validate_custom(config))
        
        # 安全验证（企业级）
        if self.validation_level == ValidationLevel.ENTERPRISE:
            all_results.extend(self._validate_security(config))
        
        # 性能影响验证（严格模式）
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.ENTERPRISE]:
            all_results.extend(self._validate_performance_impact(config))
        
        # 判断整体验证结果
        has_errors = any(r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                        for r in all_results)
        
        return not has_errors, all_results
    
    def _validate_structure(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """验证配置结构"""
        results = []
        
        required_sections = ['storage', 'processors', 'llm']
        
        for section in required_sections:
            if section not in config:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    path=section,
                    message=f"缺少必需的配置段: {section}",
                    suggestion=f"请添加 {section} 配置段"
                ))
        
        return results
    
    def _validate_constraints(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """验证配置约束"""
        results = []
        
        for path_pattern, constraints in self.constraints.items():
            # 处理通配符路径
            if '*' in path_pattern:
                matching_paths = self._find_matching_paths(config, path_pattern)
                for actual_path in matching_paths:
                    value = self._get_nested_value(config, actual_path)
                    if value is not None:
                        for constraint in constraints:
                            results.extend(constraint.validate(value, actual_path, config))
            else:
                value = self._get_nested_value(config, path_pattern)
                if value is not None:
                    for constraint in constraints:
                        results.extend(constraint.validate(value, path_pattern, config))
        
        return results
    
    def _validate_custom(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """执行自定义验证"""
        results = []
        
        for path, validator in self.custom_validators.items():
            try:
                value = self._get_nested_value(config, path)
                if value is not None:
                    results.extend(validator(value, path, config))
            except Exception as e:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    path=path,
                    message=f"自定义验证器执行失败: {str(e)}",
                    suggestion="请检查自定义验证器实现"
                ))
        
        return results
    
    def _validate_security(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """安全验证（企业级）"""
        results = []
        
        # 检查敏感信息
        sensitive_patterns = [
            (r'password', '不应在配置中存储明文密码'),
            (r'secret', '不应在配置中存储明文密钥'),
            (r'key.*[a-zA-Z0-9]{20,}', 'API密钥可能暴露')
        ]
        
        config_str = json.dumps(config, default=str).lower()
        
        for pattern, message in sensitive_patterns:
            if re.search(pattern, config_str):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    path="security",
                    message=message,
                    suggestion="请使用环境变量或密钥管理系统",
                    impact="可能存在安全风险"
                ))
        
        return results
    
    def _validate_performance_impact(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """性能影响验证"""
        results = []
        
        # 检查可能影响性能的配置
        performance_checks = [
            ("processors.ocr_processor.config.batch_size", 100, "OCR批处理大小过大可能影响内存"),
            ("llm.settings.max_tokens", 8000, "LLM最大tokens过大可能影响响应时间"),
            ("vector_db.document_store.embedding_dim", 1536, "嵌入维度过大可能影响检索速度")
        ]
        
        for path, threshold, message in performance_checks:
            value = self._get_nested_value(config, path)
            if value and isinstance(value, (int, float)) and value > threshold:
                results.append(ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    path=path,
                    message=f"{message}: 当前值 {value}",
                    suggestion=f"建议值不超过 {threshold}",
                    impact="可能影响系统性能"
                ))
        
        return results
    
    def _find_matching_paths(self, config: Dict[str, Any], pattern: str) -> List[str]:
        """查找匹配通配符模式的路径"""
        paths = []
        
        def traverse(current: Dict[str, Any], current_path: str = ""):
            for key, value in current.items():
                new_path = f"{current_path}.{key}" if current_path else key
                
                # 检查是否匹配模式
                if self._path_matches_pattern(new_path, pattern):
                    paths.append(new_path)
                
                # 递归遍历
                if isinstance(value, dict):
                    traverse(value, new_path)
        
        traverse(config)
        return paths
    
    def _path_matches_pattern(self, path: str, pattern: str) -> bool:
        """检查路径是否匹配模式"""
        # 将模式转换为正则表达式
        regex_pattern = pattern.replace('.', r'\.').replace('*', r'[^.]+')
        return re.match(f"^{regex_pattern}$", path) is not None
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """获取嵌套配置值"""
        keys = path.split('.')
        current = config
        
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        
        return current
    
    def generate_validation_report(self, results: List[ValidationResult]) -> str:
        """生成验证报告"""
        if not results:
            return "✅ 配置验证通过，未发现问题。"
        
        # 按严重程度分组
        by_severity = {}
        for result in results:
            severity = result.severity.value
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(result)
        
        report_lines = ["# 配置验证报告\n"]
        
        # 总览
        total = len(results)
        errors = len(by_severity.get('error', []))
        warnings = len(by_severity.get('warning', []))
        
        if errors > 0:
            report_lines.append(f"❌ 发现 {errors} 个错误，{warnings} 个警告")
        elif warnings > 0:
            report_lines.append(f"⚠️ 发现 {warnings} 个警告")
        else:
            report_lines.append("✅ 验证通过")
        
        report_lines.append("")
        
        # 详细结果
        for severity in ['critical', 'error', 'warning', 'info']:
            if severity in by_severity:
                severity_results = by_severity[severity]
                
                if severity == 'critical':
                    icon = "🚨"
                elif severity == 'error':
                    icon = "❌"
                elif severity == 'warning':
                    icon = "⚠️"
                else:
                    icon = "ℹ️"
                
                report_lines.append(f"## {icon} {severity.upper()} ({len(severity_results)})")
                report_lines.append("")
                
                for result in severity_results:
                    report_lines.append(f"**路径**: `{result.path}`")
                    report_lines.append(f"**问题**: {result.message}")
                    
                    if result.suggestion:
                        report_lines.append(f"**建议**: {result.suggestion}")
                    
                    if result.impact:
                        report_lines.append(f"**影响**: {result.impact}")
                    
                    report_lines.append("")
        
        return "\n".join(report_lines)

# 便捷函数
def validate_config_file(config_path: str, validation_level: ValidationLevel = ValidationLevel.STANDARD) -> Tuple[bool, str]:
    """验证配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        validator = RuntimeConfigValidator(validation_level)
        is_valid, results = validator.validate_config(config)
        report = validator.generate_validation_report(results)
        
        return is_valid, report
        
    except FileNotFoundError:
        return False, f"配置文件不存在: {config_path}"
    except json.JSONDecodeError as e:
        return False, f"配置文件JSON格式错误: {str(e)}"
    except Exception as e:
        return False, f"验证过程中发生错误: {str(e)}"

def create_constraint_example():
    """创建约束示例"""
    # 自定义验证器示例
    def validate_api_url(value: Any, path: str, config: Dict[str, Any]) -> List[ValidationResult]:
        results = []
        
        if not isinstance(value, str):
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                path=path,
                message="API URL必须是字符串",
                suggestion="请使用字符串格式的URL"
            ))
            return results
        
        if not value.startswith(('http://', 'https://')):
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                path=path,
                message="API URL建议使用HTTPS协议",
                suggestion="请使用 https:// 开头的URL以确保安全",
                impact="HTTP协议可能存在安全风险"
            ))
        
        return results
    
    return validate_api_url