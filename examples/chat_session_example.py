"""
聊天会话示例脚本

该脚本演示了如何使用聊天对话CLI进行常见操作，包括：
- 创建新的聊天会话
- 发送和接收消息
- 管理会话历史
- 导出会话记录

用法：
    python chat_session_example.py [config_path]
"""

import os
import sys
import time
import argparse
import uuid
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.chat_cli import ChatCLI


def simulate_chat_conversation(chat_cli, queries):
    """模拟聊天对话"""
    for query in queries:
        print(f"\n用户: {query}")
        
        # 处理用户输入
        with patch_input():  # 防止CLI等待用户输入
            chat_cli._process_user_input(query)
        
        # 打印最后一条消息（助手回答）
        last_message = chat_cli.session.get_last_message()
        if last_message and last_message.get("role") == "assistant":
            print(f"助手: {last_message.get('content')}\n")
        
        # 短暂暂停，便于观察
        time.sleep(0.5)


class patch_input:
    """用于模拟用户输入的上下文管理器"""
    def __enter__(self):
        self._original_input = __builtins__["input"]
        __builtins__["input"] = lambda _: "exit"
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        __builtins__["input"] = self._original_input


def run_examples(config_path):
    """运行聊天会话示例"""
    print("\n" + "=" * 80)
    print("聊天会话示例")
    print("=" * 80 + "\n")
    
    # 示例1：创建新会话
    print("示例1：创建新会话")
    print("-" * 40)
    
    # 生成唯一会话ID
    session_id = str(uuid.uuid4())
    
    # 创建CLI实例
    cli = ChatCLI(config_path=config_path, session_id=session_id, verbose=True)
    
    print(f"已创建新会话，ID: {session_id}")
    print("-" * 80 + "\n")
    time.sleep(1)  # 暂停，便于观察输出
    
    # 示例2：模拟对话
    print("示例2：模拟对话")
    print("-" * 40)
    
    # 定义示例查询
    example_queries = [
        "你好，请介绍一下这个系统的功能",
        "系统中有哪些文档？",
        "如何使用检索功能？",
        "谢谢你的帮助"
    ]
    
    # 模拟对话
    simulate_chat_conversation(cli, example_queries)
    
    print("对话示例已完成")
    print("-" * 80 + "\n")
    time.sleep(1)  # 暂停，便于观察输出
    
    # 示例3：查看会话历史
    print("示例3：查看会话历史")
    print("-" * 40)
    
    messages = cli.session.get_messages()
    
    print(f"会话 {session_id} 的历史记录:")
    for msg in messages:
        if msg.get("role") != "system":  # 跳过系统消息
            print(f"[{cli._format_timestamp(msg.get('timestamp'))}] {msg.get('role').upper()}: {msg.get('content')[:50]}...")
    
    print("-" * 80 + "\n")
    time.sleep(1)  # 暂停，便于观察输出
    
    # 示例4：导出会话记录
    print("示例4：导出会话记录")
    print("-" * 40)
    
    # 手动保存会话，确保数据已写入
    cli._save_session()
    
    # 临时修改会话ID，用于演示目的
    original_session_id = cli.session_id
    cli.session_id = session_id
    
    # 模拟命令行参数
    sys.argv = ["chat_cli.py", "--export", session_id]
    
    with patch_input():  # 防止CLI等待用户输入
        # 运行CLI的导出功能
        cli._export_session(session_id)
    
    # 恢复原始会话ID
    cli.session_id = original_session_id
    
    print("-" * 80 + "\n")
    
    # 示例5：列出所有会话
    print("示例5：列出所有会话")
    print("-" * 40)
    
    # 模拟命令行参数
    sys.argv = ["chat_cli.py", "--list"]
    
    with patch_input():  # 防止CLI等待用户输入
        # 运行CLI的列表功能
        cli._list_sessions()
    
    print("-" * 80 + "\n")
    
    print("所有示例已完成运行")
    print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="聊天会话示例脚本")
    parser.add_argument("config_path", nargs="?", default="./config/config.json", 
                        help="配置文件路径")
    args = parser.parse_args()
    
    run_examples(args.config_path)


if __name__ == "__main__":
    main()
