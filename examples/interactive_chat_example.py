"""
交互式聊天示例脚本

该脚本演示了如何创建增强的交互式聊天体验，包括格式化输出、聊天历史管理等功能。

用法：
    python interactive_chat_example.py [config_path]
"""

import os
import sys
import time
import argparse
import uuid
import readline  # 用于命令行历史记录和编辑功能
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.chat_cli import ChatCLI


class EnhancedChatInterface:
    """增强的聊天界面类"""
    
    def __init__(self, config_path):
        """
        初始化增强的聊天界面
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.session_id = str(uuid.uuid4())
        self.cli = ChatCLI(config_path=config_path, session_id=self.session_id)
        
        # 颜色代码
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'user': '\033[94m',  # 蓝色
            'assistant': '\033[92m',  # 绿色
            'system': '\033[93m',  # 黄色
            'error': '\033[91m',  # 红色
            'time': '\033[90m',  # 灰色
            'header': '\033[95m'  # 紫色
        }
    
    def print_welcome(self):
        """打印欢迎信息"""
        print(f"\n{self.colors['header']}{self.colors['bold']}{'=' * 80}{self.colors['reset']}")
        print(f"{self.colors['header']}{self.colors['bold']}欢迎使用增强型聊天界面{self.colors['reset']}")
        print(f"{self.colors['header']}{self.colors['bold']}{'=' * 80}{self.colors['reset']}")
        print(f"{self.colors['system']}会话ID: {self.session_id}{self.colors['reset']}")
        print(f"{self.colors['system']}当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{self.colors['reset']}")
        print(f"{self.colors['system']}输入 'help' 或 '?' 查看帮助信息{self.colors['reset']}")
        print(f"{self.colors['header']}{'-' * 80}{self.colors['reset']}\n")
        
        # 显示系统提示
        system_message = "我是一个基于学术文献内容的智能问答助手。您可以询问有关已上传文献的问题，我将尝试根据文献内容提供回答。"
        print(f"{self.colors['system']}系统: {system_message}{self.colors['reset']}\n")
    
    def print_help(self):
        """打印帮助信息"""
        print(f"\n{self.colors['header']}{self.colors['bold']}{'=' * 80}{self.colors['reset']}")
        print(f"{self.colors['header']}{self.colors['bold']}帮助信息{self.colors['reset']}")
        print(f"{self.colors['header']}{self.colors['bold']}{'=' * 80}{self.colors['reset']}")
        print(f"{self.colors['system']}可用命令:{self.colors['reset']}")
        print(f"{self.colors['system']}  help 或 ? - 显示此帮助信息{self.colors['reset']}")
        print(f"{self.colors['system']}  clear - 清屏{self.colors['reset']}")
        print(f"{self.colors['system']}  history [n] - 显示最近n条聊天历史{self.colors['reset']}")
        print(f"{self.colors['system']}  save - 保存当前会话{self.colors['reset']}")
        print(f"{self.colors['system']}  export - 导出会话记录{self.colors['reset']}")
        print(f"{self.colors['system']}  exit 或 quit - 退出聊天{self.colors['reset']}")
        print(f"{self.colors['header']}{'-' * 80}{self.colors['reset']}\n")
    
    def handle_history_command(self, args):
        """处理history命令"""
        # 解析参数
        try:
            count = int(args[0]) if args else 5
        except ValueError:
            count = 5
        
        # 获取消息历史
        messages = self.cli.session.get_messages()
        
        # 过滤系统消息
        messages = [msg for msg in messages if msg.get("role") != "system"]
        
        # 限制数量
        messages = messages[-count:] if count < len(messages) else messages
        
        if not messages:
            print(f"{self.colors['system']}没有聊天历史记录{self.colors['reset']}")
            return
        
        print(f"\n{self.colors['header']}最近 {len(messages)} 条聊天记录:{self.colors['reset']}")
        
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = self.cli._format_timestamp(msg.get("timestamp", ""))
            
            role_color = self.colors.get(role, self.colors['reset'])
            
            print(f"{self.colors['time']}[{timestamp}]{self.colors['reset']} {role_color}{role.upper()}:{self.colors['reset']}")
            print(f"{role_color}{content}{self.colors['reset']}\n")
    
    def run(self):
        """运行增强的聊天界面"""
        # 打印欢迎信息
        self.print_welcome()
        
        # 聊天循环
        while True:
            try:
                # 显示带颜色的提示符
                user_input = input(f"{self.colors['user']}您: {self.colors['reset']}")
            except EOFError:
                break
            except KeyboardInterrupt:
                print(f"\n{self.colors['system']}再见！{self.colors['reset']}")
                break
            
            # 去除首尾空白
            user_input = user_input.strip()
            
            # 检查退出命令
            if user_input.lower() in ["exit", "quit", "bye", "再见"]:
                print(f"\n{self.colors['system']}再见！{self.colors['reset']}")
                break
            
            # 忽略空输入
            if not user_input:
                continue
            
            # 处理特殊命令
            if user_input.lower() in ["help", "?", "帮助"]:
                self.print_help()
                continue
            
            if user_input.lower() in ["clear", "cls", "清屏"]:
                os.system("cls" if os.name == "nt" else "clear")
                self.print_welcome()
                continue
            
            if user_input.lower().startswith("history"):
                args = user_input.split()[1:] if len(user_input.split()) > 1 else []
                self.handle_history_command(args)
                continue
            
            if user_input.lower() in ["save", "保存"]:
                self.cli._save_session()
                print(f"{self.colors['system']}会话已保存{self.colors['reset']}")
                continue
            
            if user_input.lower() in ["export", "导出"]:
                self.cli._export_session(self.session_id)
                continue
            
            # 处理常规聊天消息
            self.handle_chat_message(user_input)
    
    def handle_chat_message(self, message):
        """
        处理聊天消息
        
        Args:
            message: 用户消息内容
        """
        try:
            # 添加用户消息到会话
            self.cli.session.add_message(role="user", content=message)
            
            # 在实际应用中，应调用RAG管道处理查询
            # 这里使用模拟回答和字符打印效果
            print(f"\n{self.colors['assistant']}助手: {self.colors['reset']}", end="", flush=True)
            
            # 生成模拟回答
            response = self.cli._generate_mock_response(message)
            
            # 模拟打字效果
            for char in response:
                print(char, end="", flush=True)
                time.sleep(0.01)
            print("\n")
            
            # 添加助手消息到会话
            self.cli.session.add_message(role="assistant", content=response)
            
            # 保存会话
            self.cli._save_session()
            
        except Exception as e:
            print(f"\n{self.colors['error']}处理您的问题时出错: {str(e)}{self.colors['reset']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="交互式聊天示例脚本")
    parser.add_argument("config_path", nargs="?", default="./config/config.json", 
                        help="配置文件路径")
    args = parser.parse_args()
    
    # 运行增强的聊天界面
    chat_interface = EnhancedChatInterface(args.config_path)
    chat_interface.run()


if __name__ == "__main__":
    main()
