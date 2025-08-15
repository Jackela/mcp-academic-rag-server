"""
测试清理工具 - Python版本的资源管理和清理机制

根治路径：测试内部严格 teardown，外部会话级一键回收。
分析：遗留来源包括HTTP服务器、数据库连接、消息队列、子进程、定时器、文件监听器等。
"""

import asyncio
import subprocess
import threading
import time
import signal
import os
import logging
from typing import List, Callable, Union, Any, Optional
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)

# 清理函数类型定义
CleanupFunc = Callable[[], Union[None, Any]]
AsyncCleanupFunc = Callable[[], Union[None, Any]]

class ResourceCleaner:
    """统一资源清理器"""
    
    def __init__(self):
        self.sync_cleaners: List[CleanupFunc] = []
        self.async_cleaners: List[AsyncCleanupFunc] = []
        self.processes: List[subprocess.Popen] = []
        self.threads: List[threading.Thread] = []
        self.timers: List[Any] = []
        self.servers: List[Any] = []
        self.connections: List[Any] = []
        self.temp_files: List[str] = []
        self.temp_dirs: List[str] = []
        
    def register_cleanup(self, func: CleanupFunc):
        """注册同步清理函数"""
        self.sync_cleaners.append(func)
        
    def register_async_cleanup(self, func: AsyncCleanupFunc):
        """注册异步清理函数"""
        self.async_cleaners.append(func)
        
    def register_process(self, process: subprocess.Popen):
        """注册需要清理的进程"""
        self.processes.append(process)
        
    def register_thread(self, thread: threading.Thread):
        """注册需要清理的线程"""
        self.threads.append(thread)
        
    def register_timer(self, timer: Any):
        """注册需要清理的定时器"""
        self.timers.append(timer)
        
    def register_server(self, server: Any):
        """注册需要清理的服务器"""
        self.servers.append(server)
        
    def register_connection(self, connection: Any):
        """注册需要清理的连接"""
        self.connections.append(connection)
        
    def register_temp_file(self, filepath: str):
        """注册需要清理的临时文件"""
        self.temp_files.append(filepath)
        
    def register_temp_dir(self, dirpath: str):
        """注册需要清理的临时目录"""
        self.temp_dirs.append(dirpath)

    def cleanup_processes(self):
        """清理所有注册的进程"""
        for process in self.processes:
            try:
                if process.poll() is None:  # 进程仍在运行
                    # 优雅终止
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # 强制杀死
                        process.kill()
                        process.wait()
                    logger.info(f"清理进程 PID: {process.pid}")
            except Exception as e:
                logger.warning(f"清理进程失败: {e}")
        self.processes.clear()

    def cleanup_threads(self):
        """清理所有注册的线程"""
        for thread in self.threads:
            try:
                if thread.is_alive():
                    # Python线程无法强制终止，只能等待
                    thread.join(timeout=5)
                    if thread.is_alive():
                        logger.warning(f"线程 {thread.name} 未能在5秒内结束")
            except Exception as e:
                logger.warning(f"清理线程失败: {e}")
        self.threads.clear()

    def cleanup_timers(self):
        """清理所有定时器"""
        for timer in self.timers:
            try:
                if hasattr(timer, 'cancel'):
                    timer.cancel()
                elif hasattr(timer, 'stop'):
                    timer.stop()
                elif hasattr(timer, 'close'):
                    timer.close()
            except Exception as e:
                logger.warning(f"清理定时器失败: {e}")
        self.timers.clear()

    def cleanup_servers(self):
        """清理所有服务器"""
        for server in self.servers:
            try:
                if hasattr(server, 'close'):
                    server.close()
                elif hasattr(server, 'shutdown'):
                    server.shutdown()
                elif hasattr(server, 'stop'):
                    server.stop()
            except Exception as e:
                logger.warning(f"清理服务器失败: {e}")
        self.servers.clear()

    def cleanup_connections(self):
        """清理所有连接"""
        for conn in self.connections:
            try:
                if hasattr(conn, 'close'):
                    conn.close()
                elif hasattr(conn, 'disconnect'):
                    conn.disconnect()
                elif hasattr(conn, 'quit'):
                    conn.quit()
            except Exception as e:
                logger.warning(f"清理连接失败: {e}")
        self.connections.clear()

    def cleanup_temp_files(self):
        """清理临时文件和目录"""
        import shutil
        
        # 清理临时文件
        for filepath in self.temp_files:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.debug(f"删除临时文件: {filepath}")
            except Exception as e:
                logger.warning(f"删除临时文件失败 {filepath}: {e}")
        self.temp_files.clear()
        
        # 清理临时目录
        for dirpath in self.temp_dirs:
            try:
                if os.path.exists(dirpath):
                    shutil.rmtree(dirpath)
                    logger.debug(f"删除临时目录: {dirpath}")
            except Exception as e:
                logger.warning(f"删除临时目录失败 {dirpath}: {e}")
        self.temp_dirs.clear()

    async def cleanup_async(self):
        """执行异步清理"""
        for func in self.async_cleaners:
            try:
                result = func()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"异步清理函数失败: {e}")
        self.async_cleaners.clear()

    def cleanup_sync(self):
        """执行同步清理"""
        for func in self.sync_cleaners:
            try:
                func()
            except Exception as e:
                logger.warning(f"同步清理函数失败: {e}")
        self.sync_cleaners.clear()

    async def run_full_cleanup(self):
        """执行完整清理"""
        logger.info("开始资源清理...")
        
        # 1. 执行用户注册的清理函数
        await self.cleanup_async()
        self.cleanup_sync()
        
        # 2. 清理系统资源
        self.cleanup_connections()
        self.cleanup_servers()
        self.cleanup_processes()
        self.cleanup_timers()
        self.cleanup_threads()
        
        # 3. 清理临时文件
        self.cleanup_temp_files()
        
        logger.info("资源清理完成")

    def cleanup(self):
        """同步版本的完整清理"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果事件循环正在运行，创建任务
                loop.create_task(self.run_full_cleanup())
            else:
                # 如果事件循环未运行，直接运行
                loop.run_until_complete(self.run_full_cleanup())
        except RuntimeError:
            # 如果没有事件循环，只执行同步清理
            self.cleanup_sync()
            self.cleanup_connections()
            self.cleanup_servers()
            self.cleanup_processes()
            self.cleanup_timers()
            self.cleanup_threads()
            self.cleanup_temp_files()

# 全局清理器实例
_global_cleaner = ResourceCleaner()

# 便捷函数
def register_cleanup(func: CleanupFunc):
    """注册清理函数"""
    _global_cleaner.register_cleanup(func)

def register_async_cleanup(func: AsyncCleanupFunc):
    """注册异步清理函数"""
    _global_cleaner.register_async_cleanup(func)

def register_process(process: subprocess.Popen):
    """注册进程"""
    _global_cleaner.register_process(process)

def register_thread(thread: threading.Thread):
    """注册线程"""
    _global_cleaner.register_thread(thread)

def register_server(server: Any):
    """注册服务器"""
    _global_cleaner.register_server(server)

def register_connection(connection: Any):
    """注册连接"""
    _global_cleaner.register_connection(connection)

def register_temp_file(filepath: str):
    """注册临时文件"""
    _global_cleaner.register_temp_file(filepath)

def register_temp_dir(dirpath: str):
    """注册临时目录"""
    _global_cleaner.register_temp_dir(dirpath)

async def run_cleanups():
    """运行所有清理操作"""
    await _global_cleaner.run_full_cleanup()

def cleanup_sync():
    """同步运行清理操作"""
    _global_cleaner.cleanup()

@contextmanager
def managed_resource(resource, cleanup_func=None):
    """上下文管理器，自动清理资源"""
    try:
        if cleanup_func:
            register_cleanup(lambda: cleanup_func(resource))
        else:
            # 尝试常见的清理方法
            if hasattr(resource, 'close'):
                register_cleanup(resource.close)
            elif hasattr(resource, 'cleanup'):
                register_cleanup(resource.cleanup)
            elif hasattr(resource, 'stop'):
                register_cleanup(resource.stop)
        
        yield resource
    finally:
        if cleanup_func:
            cleanup_func(resource)
        elif hasattr(resource, 'close'):
            resource.close()
        elif hasattr(resource, 'cleanup'):
            resource.cleanup()
        elif hasattr(resource, 'stop'):
            resource.stop()

def kill_port(port: int):
    """杀死占用指定端口的进程"""
    try:
        if os.name == 'nt':  # Windows
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
                        logger.info(f"杀死端口 {port} 上的进程 PID: {pid}")
        else:  # Unix-like
            result = subprocess.run(
                ['lsof', f'-ti:{port}'], 
                capture_output=True, 
                text=True, 
                check=False
            )
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    subprocess.run(['kill', '-TERM', pid], check=False)
                    logger.info(f"杀死端口 {port} 上的进程 PID: {pid}")
    except Exception as e:
        logger.warning(f"杀死端口 {port} 进程失败: {e}")

def emergency_cleanup():
    """紧急清理 - 杀死所有相关进程"""
    logger.warning("执行紧急清理...")
    
    try:
        # 杀死常用端口的进程
        common_ports = [5000, 8000, 8080, 3000, 9000]
        for port in common_ports:
            kill_port(port)
        
        # 杀死Python相关进程（谨慎使用）
        current_pid = os.getpid()
        if os.name != 'nt':  # Unix-like系统
            subprocess.run([
                'pkill', '-f', 'python.*test'
            ], check=False)
        
        # 清理全局资源
        cleanup_sync()
        
    except Exception as e:
        logger.error(f"紧急清理失败: {e}")

# 信号处理器
def _signal_handler(signum, frame):
    """信号处理器，执行清理"""
    logger.info(f"收到信号 {signum}，执行清理...")
    emergency_cleanup()

# 注册信号处理器
if os.name != 'nt':  # Unix-like系统
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)