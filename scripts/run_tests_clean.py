#!/usr/bin/env python3
"""
增强的测试运行脚本 - 确保资源彻底清理

根治路径：测试内部严格 teardown，外部会话级一键回收。
包含：
- 测试前后的资源清理
- 进程和端口监控
- 失败时的诊断信息输出
- 内存泄漏检测
"""

import os
import sys
import subprocess
import logging
import signal
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.utils.cleanup import emergency_cleanup, kill_port

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(project_root / 'logs' / f'test_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class TestRunner:
    """增强的测试运行器"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_processes: List[subprocess.Popen] = []
        self.start_time = None
        self.ports_to_cleanup = [5000, 8000, 8080, 3000, 9000, 9229]  # 常用端口
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame):
        """信号处理器，执行清理"""
        logger.warning(f"收到信号 {signum}，执行测试清理...")
        self.cleanup_all()
        sys.exit(1)
    
    def preflight_check(self) -> bool:
        """测试前检查"""
        logger.info("=== 测试前检查 ===")
        
        try:
            # 1. 检查项目结构
            required_dirs = ['tests', 'core', 'models', 'processors']
            for dir_name in required_dirs:
                if not (self.project_root / dir_name).exists():
                    logger.error(f"缺少必要目录: {dir_name}")
                    return False
            
            # 2. 清理之前的测试残留
            self.cleanup_test_residuals()
            
            # 3. 检查内存状态
            memory_info = self.get_memory_info()
            logger.info(f"启动内存状态: {memory_info}")
            
            # 4. 检查端口占用
            occupied_ports = self.check_port_usage()
            if occupied_ports:
                logger.warning(f"检测到占用端口: {occupied_ports}")
                self.cleanup_ports()
            
            # 5. 检查Python环境
            python_version = sys.version_info
            if python_version < (3, 8):
                logger.error(f"Python版本过低: {python_version}, 需要3.8+")
                return False
            
            logger.info("✅ 测试前检查通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试前检查失败: {e}")
            return False
    
    def cleanup_test_residuals(self):
        """清理测试残留文件和进程"""
        logger.info("清理测试残留...")
        
        try:
            # 1. 清理临时文件和目录
            temp_patterns = [
                'test_*',
                'mcp_rag_test_*',
                'pytest_cache',
                '__pycache__',
                '*.pyc',
                '.coverage*',
                'htmlcov'
            ]
            
            for pattern in temp_patterns:
                cmd = f'find "{self.project_root}" -name "{pattern}" -type d -exec rm -rf {{}} + 2>/dev/null || true'
                if os.name == 'nt':  # Windows
                    cmd = f'for /d /r "{self.project_root}" %i in ({pattern}) do @if exist "%i" rmdir /s /q "%i" 2>nul'
                os.system(cmd)
            
            # 2. 杀死相关Python进程
            if os.name != 'nt':  # Unix-like
                subprocess.run(['pkill', '-f', 'python.*test'], check=False)
                subprocess.run(['pkill', '-f', 'pytest'], check=False)
            
            # 3. 清理端口
            self.cleanup_ports()
            
            logger.info("✅ 测试残留清理完成")
            
        except Exception as e:
            logger.warning(f"测试残留清理失败: {e}")
    
    def cleanup_ports(self):
        """清理占用的端口"""
        for port in self.ports_to_cleanup:
            try:
                kill_port(port)
            except Exception as e:
                logger.debug(f"清理端口 {port} 失败: {e}")
    
    def check_port_usage(self) -> List[int]:
        """检查端口占用情况"""
        occupied = []
        for port in self.ports_to_cleanup:
            try:
                if os.name == 'nt':  # Windows
                    result = subprocess.run(
                        ['netstat', '-an'], 
                        capture_output=True, 
                        text=True, 
                        check=True,
                        timeout=5
                    )
                    if f':{port}' in result.stdout and 'LISTENING' in result.stdout:
                        occupied.append(port)
                else:  # Unix-like
                    result = subprocess.run(
                        ['ss', '-tulpn'], 
                        capture_output=True, 
                        text=True, 
                        check=False,
                        timeout=5
                    )
                    if result.returncode == 0 and f':{port}' in result.stdout:
                        occupied.append(port)
            except Exception as e:
                logger.debug(f"检查端口 {port} 失败: {e}")
        
        return occupied
    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total_gb': round(memory.total / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'percent': memory.percent,
                'available_gb': round(memory.available / (1024**3), 2)
            }
        except ImportError:
            return {'error': 'psutil not available'}
        except Exception as e:
            return {'error': str(e)}
    
    def get_process_info(self) -> str:
        """获取进程信息用于诊断"""
        try:
            if os.name == 'nt':  # Windows
                result = subprocess.run(
                    ['tasklist', '/FI', 'IMAGENAME eq python.exe'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            else:  # Unix-like
                result = subprocess.run(
                    ['ps', 'aux'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            return result.stdout
        except Exception as e:
            return f"获取进程信息失败: {e}"
    
    def run_tests(self, test_args: List[str] = None) -> int:
        """运行测试"""
        logger.info("=== 开始运行测试 ===")
        self.start_time = time.time()
        
        try:
            # 构建pytest命令
            cmd = [
                sys.executable, '-m', 'pytest',
                '--tb=short',  # 简化traceback
                '--strict-markers',  # 严格标记检查
                '--strict-config',  # 严格配置检查
                '-v',  # 详细输出
                '--durations=10',  # 显示最慢的10个测试
                '--maxfail=5',  # 最多失败5个测试就停止
            ]
            
            # 添加覆盖率报告
            if '--cov' not in (test_args or []):
                cmd.extend(['--cov=.', '--cov-report=html', '--cov-report=term'])
            
            # 添加用户参数
            if test_args:
                cmd.extend(test_args)
            else:
                cmd.append('tests/')  # 默认运行所有测试
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            # 设置环境变量
            env = os.environ.copy()
            env['TESTING'] = 'true'
            env['PYTHONPATH'] = str(self.project_root)
            env['PYTHONDONTWRITEBYTECODE'] = '1'  # 不生成.pyc文件
            
            # 运行测试
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # 行缓冲
                universal_newlines=True
            )
            
            self.test_processes.append(process)
            
            # 实时输出测试结果
            for line in process.stdout:
                print(line, end='')
                if 'FAILED' in line or 'ERROR' in line:
                    logger.warning(f"测试失败: {line.strip()}")
            
            # 等待测试完成
            return_code = process.wait()
            
            duration = time.time() - self.start_time
            logger.info(f"测试完成，耗时: {duration:.2f}秒，返回码: {return_code}")
            
            return return_code
            
        except KeyboardInterrupt:
            logger.warning("用户中断测试")
            return 1
        except Exception as e:
            logger.error(f"运行测试失败: {e}")
            return 1
    
    def postflight_check(self, return_code: int):
        """测试后检查"""
        logger.info("=== 测试后检查 ===")
        
        try:
            # 1. 检查内存状态
            memory_info = self.get_memory_info()
            logger.info(f"结束内存状态: {memory_info}")
            
            # 2. 检查是否有遗留进程
            if return_code != 0:
                logger.error("测试失败，输出诊断信息:")
                logger.error(f"进程信息:\n{self.get_process_info()}")
                
                # 检查端口
                occupied_ports = self.check_port_usage()
                if occupied_ports:
                    logger.error(f"遗留端口: {occupied_ports}")
            
            # 3. 生成测试报告摘要
            self.generate_test_summary(return_code)
            
        except Exception as e:
            logger.warning(f"测试后检查失败: {e}")
    
    def generate_test_summary(self, return_code: int):
        """生成测试摘要"""
        try:
            duration = time.time() - self.start_time if self.start_time else 0
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': round(duration, 2),
                'return_code': return_code,
                'status': 'PASSED' if return_code == 0 else 'FAILED',
                'memory_info': self.get_memory_info(),
                'occupied_ports': self.check_port_usage()
            }
            
            # 保存摘要到文件
            summary_file = self.project_root / 'test-results' / 'last_run_summary.json'
            summary_file.parent.mkdir(exist_ok=True)
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"测试摘要已保存: {summary_file}")
            
        except Exception as e:
            logger.warning(f"生成测试摘要失败: {e}")
    
    def cleanup_all(self):
        """清理所有资源"""
        logger.info("执行最终清理...")
        
        try:
            # 1. 终止测试进程
            for process in self.test_processes:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
            
            # 2. 执行紧急清理
            emergency_cleanup()
            
            # 3. 清理端口
            self.cleanup_ports()
            
            logger.info("✅ 最终清理完成")
            
        except Exception as e:
            logger.error(f"最终清理失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强的测试运行器')
    parser.add_argument('--clean', action='store_true', help='仅执行清理，不运行测试')
    parser.add_argument('--preflight-only', action='store_true', help='仅执行预检查')
    parser.add_argument('--no-coverage', action='store_true', help='禁用覆盖率报告')
    parser.add_argument('test_args', nargs='*', help='传递给pytest的参数')
    
    args = parser.parse_args()
    
    # 创建测试运行器
    runner = TestRunner(project_root)
    
    try:
        if args.clean:
            logger.info("执行清理模式...")
            runner.cleanup_test_residuals()
            runner.cleanup_all()
            return 0
        
        # 预检查
        if not runner.preflight_check():
            logger.error("预检查失败，退出")
            return 1
        
        if args.preflight_only:
            logger.info("仅执行预检查，完成")
            return 0
        
        # 准备测试参数
        test_args = args.test_args or []
        if args.no_coverage:
            test_args = [arg for arg in test_args if not arg.startswith('--cov')]
        
        # 运行测试
        return_code = runner.run_tests(test_args)
        
        # 后检查
        runner.postflight_check(return_code)
        
        return return_code
        
    except KeyboardInterrupt:
        logger.warning("用户中断")
        return 1
    except Exception as e:
        logger.error(f"运行失败: {e}")
        return 1
    finally:
        # 确保资源清理
        runner.cleanup_all()

if __name__ == '__main__':
    sys.exit(main())