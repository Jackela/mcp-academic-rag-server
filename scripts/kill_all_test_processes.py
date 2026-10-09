#!/usr/bin/env python3
"""
彻底清理所有测试相关进程
包括Node.js、Python测试进程、端口占用等
"""

import logging
import os
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def kill_processes_by_name(process_names):
    """根据进程名杀死进程"""
    for process_name in process_names:
        try:
            if os.name == "nt":  # Windows
                result = subprocess.run(
                    ["taskkill", "/F", "/IM", process_name], capture_output=True, text=True, check=False
                )
                if result.returncode == 0:
                    logger.info(f"✅ 已杀死 {process_name} 进程")
                else:
                    logger.debug(f"📋 没有找到 {process_name} 进程")
            else:  # Unix-like
                subprocess.run(["pkill", "-f", process_name], check=False)
                logger.info(f"✅ 已杀死 {process_name} 进程")
        except Exception as e:
            logger.debug(f"杀死 {process_name} 失败: {e}")


def kill_processes_by_port(ports):
    """根据端口杀死进程"""
    for port in ports:
        try:
            if os.name == "nt":  # Windows
                # 查找占用端口的进程
                result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, check=True)

                for line in result.stdout.split("\n"):
                    if f":{port}" in line and "LISTENING" in line:
                        parts = line.split()
                        if parts:
                            pid = parts[-1]
                            try:
                                subprocess.run(["taskkill", "/F", "/PID", pid], check=True)
                                logger.info(f"✅ 已杀死占用端口 {port} 的进程 PID: {pid}")
                            except subprocess.CalledProcessError:
                                logger.debug(f"无法杀死进程 PID: {pid}")
            else:  # Unix-like
                result = subprocess.run(["lsof", f"-ti:{port}"], capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    pids = result.stdout.strip().split("\n")
                    for pid in pids:
                        if pid:
                            subprocess.run(["kill", "-TERM", pid], check=False)
                            logger.info(f"✅ 已杀死占用端口 {port} 的进程 PID: {pid}")
        except Exception as e:
            logger.debug(f"清理端口 {port} 失败: {e}")


def main():
    """主清理函数"""
    logger.info("🧹 开始彻底清理测试环境...")

    # 1. 杀死Node.js进程 (MCP Inspector)
    logger.info("🔧 清理Node.js进程...")
    kill_processes_by_name(["node.exe", "node"])

    # 2. 杀死Python测试进程
    logger.info("🐍 清理Python测试进程...")
    kill_processes_by_name(["python.exe"])

    # 3. 清理特定端口
    logger.info("🌐 清理端口占用...")
    test_ports = [6274, 6277, 5000, 8000, 8080, 3000, 9000, 9229]
    kill_processes_by_port(test_ports)

    # 4. 等待进程完全退出
    logger.info("⏱️ 等待进程清理完成...")
    time.sleep(2)

    # 5. 验证清理结果
    logger.info("🔍 验证清理结果...")
    try:
        if os.name == "nt":
            # 检查Node.js进程
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq node.exe"], capture_output=True, text=True, check=False
            )
            node_processes = [line for line in result.stdout.split("\n") if "node.exe" in line]

            if node_processes:
                logger.warning(f"⚠️ 仍有 {len(node_processes)} 个Node.js进程运行")
            else:
                logger.info("✅ 无Node.js进程残留")

            # 检查Python进程
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq python.exe"], capture_output=True, text=True, check=False
            )
            python_processes = [
                line for line in result.stdout.split("\n") if "python.exe" in line and ("test" in line or "mcp" in line)
            ]

            if python_processes:
                logger.warning(f"⚠️ 仍有 {len(python_processes)} 个Python测试进程运行")
            else:
                logger.info("✅ 无Python测试进程残留")

    except Exception as e:
        logger.warning(f"验证清理结果失败: {e}")

    logger.info("🎉 测试环境清理完成")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("用户中断清理")
        sys.exit(1)
    except Exception as e:
        logger.error(f"清理失败: {e}")
        sys.exit(1)
