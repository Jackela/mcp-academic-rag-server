#!/usr/bin/env python3
"""
测试环境清理脚本 - 确保测试后彻底清理资源

完整清理测试残留的进程、端口、临时文件等，避免Git Bash不退出问题。
用于Claude Code会话间的环境重置。
"""

import os
import sys
import subprocess
import shutil
import tempfile
import logging
import signal
import time
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResourceCleaner:
    """资源清理器 - 确保彻底清理"""
    
    def __init__(self):
        self.cleaned_pids = []
        self.cleaned_ports = []

def cleanup_processes():
    """清理Python测试进程"""
    logger.info("清理Python测试进程...")
    
    try:
        if os.name == 'nt':  # Windows
            # 杀死pytest相关进程
            subprocess.run(['taskkill', '/F', '/IM', 'python.exe', '/FI', 'WINDOWTITLE eq pytest*'], 
                         check=False, capture_output=True)
        else:  # Unix-like
            # 杀死pytest和测试相关进程
            subprocess.run(['pkill', '-f', 'pytest'], check=False)
            subprocess.run(['pkill', '-f', 'python.*test'], check=False)
            subprocess.run(['pkill', '-f', 'mcp.*test'], check=False)
        
        logger.info("✅ 进程清理完成")
    except Exception as e:
        logger.warning(f"进程清理失败: {e}")

def cleanup_ports():
    """清理常用端口"""
    ports = [5000, 8000, 8080, 3000, 9000, 9229]
    logger.info(f"清理端口: {ports}")
    
    for port in ports:
        try:
            if os.name == 'nt':  # Windows
                # 查找占用端口的进程
                result = subprocess.run(
                    ['netstat', '-ano'], 
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                for line in result.stdout.split('\n'):
                    if f':{port}' in line and 'LISTENING' in line:
                        parts = line.split()
                        if parts:
                            pid = parts[-1]
                            subprocess.run(['taskkill', '/F', '/PID', pid], check=False)
                            logger.info(f"杀死端口 {port} 进程 PID: {pid}")
            else:  # Unix-like
                result = subprocess.run(
                    ['lsof', f'-ti:{port}'], 
                    capture_output=True, 
                    text=True, 
                    check=False
                )
                if result.returncode == 0:
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        if pid:
                            subprocess.run(['kill', '-TERM', pid], check=False)
                            logger.info(f"杀死端口 {port} 进程 PID: {pid}")
        except Exception as e:
            logger.debug(f"清理端口 {port} 失败: {e}")
    
    logger.info("✅ 端口清理完成")

def cleanup_temp_files():
    """清理临时文件和目录"""
    logger.info("清理临时文件...")
    
    try:
        # 1. 清理项目内的测试文件
        patterns_to_remove = [
            'test_*',
            'mcp_rag_test_*',
            'pytest_cache',
            '__pycache__',
            '.pytest_cache',
            '.coverage*',
            'htmlcov',
            'temp',
            '*.pyc'
        ]
        
        for pattern in patterns_to_remove:
            try:
                if os.name == 'nt':  # Windows
                    # 删除文件
                    subprocess.run(f'del /s /q "{project_root}\\{pattern}" 2>nul', shell=True, check=False)
                    # 删除目录
                    subprocess.run(f'for /d /r "{project_root}" %i in ({pattern}) do @if exist "%i" rmdir /s /q "%i" 2>nul', shell=True, check=False)
                else:  # Unix-like
                    # 删除文件和目录
                    subprocess.run(f'find "{project_root}" -name "{pattern}" -exec rm -rf {{}} + 2>/dev/null', shell=True, check=False)
            except Exception as e:
                logger.debug(f"清理模式 {pattern} 失败: {e}")
        
        # 2. 清理系统临时目录中的测试文件
        temp_dir = Path(tempfile.gettempdir())
        for item in temp_dir.glob('mcp_*'):
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
                logger.debug(f"删除临时文件: {item}")
            except Exception as e:
                logger.debug(f"删除临时文件失败 {item}: {e}")
        
        # 3. 清理特定的测试目录
        test_dirs = [
            project_root / 'test_data',
            project_root / 'test_output',
            project_root / 'temp',
            project_root / 'logs' / 'test_*.log'
        ]
        
        for test_dir in test_dirs:
            try:
                if test_dir.exists():
                    if test_dir.is_dir():
                        shutil.rmtree(test_dir, ignore_errors=True)
                    else:
                        test_dir.unlink(missing_ok=True)
                    logger.debug(f"删除测试目录: {test_dir}")
            except Exception as e:
                logger.debug(f"删除测试目录失败 {test_dir}: {e}")
        
        logger.info("✅ 临时文件清理完成")
    except Exception as e:
        logger.warning(f"临时文件清理失败: {e}")

def reset_git_state():
    """重置Git状态（可选）"""
    try:
        # 检查是否是Git仓库
        if (project_root / '.git').exists():
            logger.info("重置Git状态...")
            
            # 清理未跟踪的文件（测试生成的）
            subprocess.run(['git', 'clean', '-fd'], cwd=project_root, check=False, capture_output=True)
            
            # 重置任何意外的更改
            subprocess.run(['git', 'checkout', '.'], cwd=project_root, check=False, capture_output=True)
            
            logger.info("✅ Git状态重置完成")
    except Exception as e:
        logger.debug(f"Git状态重置失败: {e}")

def check_environment():
    """检查环境状态"""
    logger.info("检查环境状态...")
    
    try:
        # 检查Python进程
        if os.name == 'nt':
            result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                                  capture_output=True, text=True, check=False)
        else:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=False)
        
        python_processes = [line for line in result.stdout.split('\n') 
                          if 'python' in line.lower() and ('test' in line or 'pytest' in line)]
        
        if python_processes:
            logger.warning(f"发现 {len(python_processes)} 个Python测试进程仍在运行")
            for proc in python_processes[:3]:  # 只显示前3个
                logger.warning(f"  - {proc.strip()}")
        else:
            logger.info("✅ 无Python测试进程残留")
        
        # 检查端口占用
        ports_to_check = [5000, 8000, 8080, 3000, 9000]
        occupied_ports = []
        
        for port in ports_to_check:
            try:
                if os.name == 'nt':
                    result = subprocess.run(['netstat', '-an'], capture_output=True, text=True, check=True)
                    if f':{port}' in result.stdout and 'LISTENING' in result.stdout:
                        occupied_ports.append(port)
                else:
                    result = subprocess.run(['ss', '-tulpn'], capture_output=True, text=True, check=False)
                    if result.returncode == 0 and f':{port}' in result.stdout:
                        occupied_ports.append(port)
            except:
                pass
        
        if occupied_ports:
            logger.warning(f"端口仍被占用: {occupied_ports}")
        else:
            logger.info("✅ 无端口占用")
        
    except Exception as e:
        logger.warning(f"环境检查失败: {e}")

def main():
    """主清理函数"""
    logger.info("=== 开始测试环境清理 ===")
    
    try:
        # 1. 清理进程
        cleanup_processes()
        
        # 2. 清理端口
        cleanup_ports()
        
        # 3. 清理临时文件
        cleanup_temp_files()
        
        # 4. 重置Git状态（可选）
        if '--reset-git' in sys.argv:
            reset_git_state()
        
        # 5. 检查环境
        if '--check' in sys.argv:
            check_environment()
        
        logger.info("=== 测试环境清理完成 ===")
        
    except KeyboardInterrupt:
        logger.warning("用户中断清理")
    except Exception as e:
        logger.error(f"清理失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())