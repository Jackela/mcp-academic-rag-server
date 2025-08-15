#!/usr/bin/env python3
"""
项目结构验证脚本

验证项目重构后的结构是否符合企业级标准：
- 根目录文件数量控制
- 模块分层清晰
- 导入路径正确
- 测试结构完整
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

project_root = Path(__file__).parent.parent

def analyze_imports(file_path: Path) -> Tuple[List[str], List[str]]:
    """分析Python文件的导入"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        local_imports = []
        external_imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(('core', 'models', 'processors', 'rag', 'connectors', 'utils')):
                        local_imports.append(alias.name)
                    else:
                        external_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith(('core', 'models', 'processors', 'rag', 'connectors', 'utils')):
                    local_imports.append(node.module)
                else:
                    if node.module:
                        external_imports.append(node.module)
        
        return local_imports, external_imports
    except Exception as e:
        print(f"分析导入失败 {file_path}: {e}")
        return [], []

def validate_root_directory() -> Dict[str, any]:
    """验证根目录结构"""
    print("🔍 验证根目录结构...")
    
    root_files = list(project_root.glob("*.py"))
    root_python_count = len(root_files)
    
    # 允许的根目录Python文件
    allowed_root_files = {'app.py', 'webapp.py', '__init__.py', 'setup.py'}
    actual_root_files = {f.name for f in root_files}
    
    unexpected_files = actual_root_files - allowed_root_files
    missing_files = {'app.py', 'webapp.py', '__init__.py'} - actual_root_files
    
    result = {
        'total_python_files': root_python_count,
        'allowed_files': list(allowed_root_files),
        'actual_files': list(actual_root_files),
        'unexpected_files': list(unexpected_files),
        'missing_files': list(missing_files),
        'compliant': root_python_count <= 5 and len(unexpected_files) == 0
    }
    
    if result['compliant']:
        print(f"  ✅ 根目录Python文件: {root_python_count}/5 (合规)")
    else:
        print(f"  ❌ 根目录Python文件: {root_python_count}/5 (超标)")
        if unexpected_files:
            print(f"     意外文件: {unexpected_files}")
    
    return result

def validate_module_structure() -> Dict[str, any]:
    """验证模块结构"""
    print("🔍 验证模块结构...")
    
    expected_modules = {
        'core': ['__init__.py', 'config_manager.py', 'pipeline.py', 'server_context.py'],
        'models': ['__init__.py', 'document.py', 'process_result.py'],
        'processors': ['__init__.py', 'base_processor.py'],
        'rag': ['__init__.py', 'haystack_pipeline.py', 'prompt_builder.py'],
        'connectors': ['__init__.py', 'base_llm_connector.py'],
        'servers': ['mcp_server.py', 'mcp_server_enhanced.py'],
        'tests': ['conftest.py', 'README.md'],
        'utils': ['error_handling.py', 'logging_utils.py']
    }
    
    results = {}
    
    for module_name, expected_files in expected_modules.items():
        module_path = project_root / module_name
        if not module_path.exists():
            results[module_name] = {'exists': False, 'files': [], 'missing': expected_files}
            print(f"  ❌ 模块缺失: {module_name}")
            continue
        
        actual_files = [f.name for f in module_path.glob("*.py")]
        missing_files = [f for f in expected_files if f.endswith('.py') and f not in actual_files]
        
        results[module_name] = {
            'exists': True,
            'files': actual_files,
            'missing': missing_files,
            'compliant': len(missing_files) == 0
        }
        
        if results[module_name]['compliant']:
            print(f"  ✅ 模块 {module_name}: {len(actual_files)} 文件")
        else:
            print(f"  ⚠️  模块 {module_name}: 缺失 {missing_files}")
    
    return results

def validate_import_paths() -> Dict[str, any]:
    """验证导入路径"""
    print("🔍 验证导入路径...")
    
    python_files = list(project_root.rglob("*.py"))
    python_files = [f for f in python_files if not any(part.startswith('.') for part in f.parts)]
    
    import_issues = []
    circular_imports = []
    module_graph = defaultdict(set)
    
    for file_path in python_files:
        try:
            local_imports, external_imports = analyze_imports(file_path)
            
            # 检查循环导入
            current_module = str(file_path.relative_to(project_root)).replace('/', '.').replace('\\', '.').rstrip('.py')
            
            for imp in local_imports:
                module_graph[current_module].add(imp)
            
            # 检查导入路径是否存在
            for imp in local_imports:
                imp_parts = imp.split('.')
                imp_path = project_root / '/'.join(imp_parts)
                
                if not (imp_path.with_suffix('.py').exists() or 
                       (imp_path / '__init__.py').exists()):
                    import_issues.append({
                        'file': str(file_path.relative_to(project_root)),
                        'import': imp,
                        'issue': 'path_not_found'
                    })
        
        except Exception as e:
            import_issues.append({
                'file': str(file_path.relative_to(project_root)),
                'import': 'N/A',
                'issue': f'parse_error: {e}'
            })
    
    # 简单的循环导入检测
    def has_cycle(graph, start, visited, rec_stack):
        visited.add(start)
        rec_stack.add(start)
        
        for neighbor in graph.get(start, set()):
            if neighbor not in visited:
                if has_cycle(graph, neighbor, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                circular_imports.append(f"{start} -> {neighbor}")
                return True
        
        rec_stack.remove(start)
        return False
    
    visited = set()
    for node in module_graph:
        if node not in visited:
            has_cycle(module_graph, node, visited, set())
    
    result = {
        'total_files_checked': len(python_files),
        'import_issues': import_issues,
        'circular_imports': circular_imports,
        'compliant': len(import_issues) == 0 and len(circular_imports) == 0
    }
    
    if result['compliant']:
        print(f"  ✅ 导入路径: {len(python_files)} 文件检查通过")
    else:
        print(f"  ❌ 导入路径: {len(import_issues)} 问题, {len(circular_imports)} 循环导入")
        for issue in import_issues[:3]:  # 只显示前3个
            print(f"     {issue['file']}: {issue['issue']}")
    
    return result

def validate_test_structure() -> Dict[str, any]:
    """验证测试结构"""
    print("🔍 验证测试结构...")
    
    tests_dir = project_root / 'tests'
    if not tests_dir.exists():
        return {'exists': False, 'compliant': False}
    
    expected_test_dirs = ['unit', 'integration', 'manual', 'utils']
    actual_test_dirs = [d.name for d in tests_dir.iterdir() if d.is_dir()]
    
    test_files = list(tests_dir.rglob("test_*.py"))
    conftest_files = list(tests_dir.rglob("conftest.py"))
    
    result = {
        'exists': True,
        'test_directories': actual_test_dirs,
        'test_files_count': len(test_files),
        'conftest_files': len(conftest_files),
        'has_utils_cleanup': (tests_dir / 'utils' / 'cleanup.py').exists(),
        'compliant': len(test_files) >= 5 and len(conftest_files) >= 1
    }
    
    if result['compliant']:
        print(f"  ✅ 测试结构: {len(test_files)} 测试文件, {len(conftest_files)} conftest")
    else:
        print(f"  ❌ 测试结构: 测试文件不足或缺失conftest")
    
    if result['has_utils_cleanup']:
        print(f"  ✅ 清理工具: tests/utils/cleanup.py 存在")
    else:
        print(f"  ⚠️  清理工具: tests/utils/cleanup.py 缺失")
    
    return result

def validate_documentation() -> Dict[str, any]:
    """验证文档结构"""
    print("🔍 验证文档结构...")
    
    docs_dir = project_root / 'docs'
    readme_files = list(project_root.glob("README*.md"))
    
    essential_docs = [
        'README.md',
        'CHANGELOG.md',
        'CONTRIBUTING.md'
    ]
    
    existing_docs = [f.name for f in project_root.glob("*.md")]
    missing_docs = [doc for doc in essential_docs if doc not in existing_docs]
    
    result = {
        'docs_dir_exists': docs_dir.exists(),
        'readme_files': len(readme_files),
        'essential_docs': essential_docs,
        'existing_docs': existing_docs,
        'missing_docs': missing_docs,
        'compliant': len(missing_docs) <= 1 and len(readme_files) >= 1
    }
    
    if result['compliant']:
        print(f"  ✅ 文档结构: {len(existing_docs)} 文档文件")
    else:
        print(f"  ⚠️  文档结构: 缺失 {missing_docs}")
    
    return result

def validate_configuration() -> Dict[str, any]:
    """验证配置结构"""
    print("🔍 验证配置结构...")
    
    config_dir = project_root / 'config'
    if not config_dir.exists():
        return {'exists': False, 'compliant': False}
    
    config_files = list(config_dir.glob("*.json"))
    required_configs = ['config.json']
    
    has_examples = any('example' in f.name for f in config_files)
    has_env_configs = any(env in f.name for f in config_files for env in ['development', 'production'])
    
    result = {
        'exists': True,
        'config_files': [f.name for f in config_files],
        'has_examples': has_examples,
        'has_env_configs': has_env_configs,
        'compliant': len(config_files) >= 2 and has_examples
    }
    
    if result['compliant']:
        print(f"  ✅ 配置结构: {len(config_files)} 配置文件")
    else:
        print(f"  ⚠️  配置结构: 配置文件不足")
    
    return result

def generate_structure_report(results: Dict[str, Dict]) -> str:
    """生成结构报告"""
    
    total_checks = len(results)
    passed_checks = sum(1 for r in results.values() if r.get('compliant', False))
    
    compliance_rate = (passed_checks / total_checks) * 100
    
    report = f"""
# 项目结构验证报告

**生成时间**: {os.popen('date').read().strip()}
**合规率**: {compliance_rate:.1f}% ({passed_checks}/{total_checks})

## 检查结果摘要

| 检查项 | 状态 | 详情 |
|--------|------|------|
"""
    
    for check_name, result in results.items():
        status = "✅ 通过" if result.get('compliant', False) else "❌ 未通过"
        
        if check_name == 'root_directory':
            detail = f"{result['total_python_files']}/5 Python文件"
        elif check_name == 'module_structure':
            modules_count = sum(1 for r in result.values() if isinstance(r, dict) and r.get('exists', False))
            detail = f"{modules_count} 个模块"
        elif check_name == 'import_paths':
            detail = f"{result['total_files_checked']} 文件检查, {len(result['import_issues'])} 问题"
        elif check_name == 'test_structure':
            detail = f"{result['test_files_count']} 测试文件"
        elif check_name == 'documentation':
            detail = f"{len(result['existing_docs'])} 文档文件"
        elif check_name == 'configuration':
            detail = f"{len(result['config_files'])} 配置文件"
        else:
            detail = "详见详细报告"
        
        report += f"| {check_name} | {status} | {detail} |\n"
    
    report += f"""

## 改进建议

"""
    
    # 根据结果生成改进建议
    if not results['root_directory']['compliant']:
        report += "- 根目录Python文件过多，建议移动到合适的模块目录\n"
    
    if results['import_paths']['import_issues']:
        report += "- 存在导入路径问题，需要修复模块引用\n"
    
    if not results['test_structure']['compliant']:
        report += "- 测试结构需要完善，增加测试用例和conftest配置\n"
    
    if compliance_rate < 80:
        report += "- 项目结构合规率较低，建议优先解决标记为❌的问题\n"
    elif compliance_rate >= 90:
        report += "- 项目结构合规率良好，可以进入下一阶段开发\n"
    
    return report

def main():
    """主验证函数"""
    print("🚀 开始项目结构验证...")
    print(f"📁 项目根目录: {project_root}")
    print("="*50)
    
    # 执行所有验证
    results = {
        'root_directory': validate_root_directory(),
        'module_structure': validate_module_structure(),
        'import_paths': validate_import_paths(),
        'test_structure': validate_test_structure(),
        'documentation': validate_documentation(),
        'configuration': validate_configuration()
    }
    
    print("="*50)
    
    # 生成报告
    report = generate_structure_report(results)
    print(report)
    
    # 保存报告到文件
    report_file = project_root / 'reports' / 'structure_validation.md'
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 详细报告已保存: {report_file}")
    
    # 返回退出码
    total_checks = len(results)
    passed_checks = sum(1 for r in results.values() if r.get('compliant', False))
    compliance_rate = (passed_checks / total_checks) * 100
    
    if compliance_rate >= 80:
        print("🎉 项目结构验证通过！")
        return 0
    else:
        print("⚠️  项目结构需要改进")
        return 1

if __name__ == '__main__':
    sys.exit(main())