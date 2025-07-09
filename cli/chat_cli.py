"""
聊天对话命令行界面

该模块提供交互式聊天界面，支持基于文档内容的自然语言对话。
"""

import os
import sys
import argparse
import logging
import json
import uuid
import readline  # 用于命令行历史记录和编辑功能
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
from pathlib import Path

# 添加项目根目录到系统路径，确保能够导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_manager import ConfigManager
from rag.chat_session import ChatSession, ChatSessionManager
from rag.haystack_pipeline import RAGPipeline
from connectors.haystack_llm_connector import HaystackLLMConnector, HaystackLLMFactory


class ChatCLI:
    """聊天对话命令行界面类"""
    
    def __init__(
        self, 
        config_path: str = "./config/config.json",
        session_id: str = None,
        verbose: bool = False
    ):
        """
        初始化聊天对话命令行界面
        
        Args:
            config_path (str): 配置文件路径
            session_id (str, optional): 会话ID，如不指定则自动生成
            verbose (bool): 是否显示详细日志
        """
        self.config_path = config_path
        self.session_id = session_id or str(uuid.uuid4())
        self.verbose = verbose
        
        # 设置日志级别
        self.log_level = logging.DEBUG if verbose else logging.INFO
        
        # 初始化组件
        self._init_components()
    
    def _init_components(self):
        """初始化组件：配置管理器、RAG系统、聊天会话等"""
        try:
            # 初始化配置管理器
            self.config_manager = ConfigManager(self.config_path)
            
            # 设置日志
            self._setup_logging()
            
            # 创建数据目录
            self._create_data_dirs()
            
            # 记录初始化信息
            self.logger.info(f"聊天对话CLI初始化完成，会话ID：{self.session_id}")
            
            # 初始化聊天会话
            # 在实际应用中，应初始化RAG管道和聊天会话
            # 由于RAG系统需要多个组件，这里暂时使用模拟会话
            self.session = self._create_mock_session()
            
        except Exception as e:
            print(f"初始化聊天对话CLI失败: {str(e)}")
            sys.exit(1)
    
    def _setup_logging(self):
        """设置日志系统"""
        log_config = self.config_manager.get_value("logging", {})
        log_level_name = log_config.get("level", "INFO")
        
        if self.verbose:
            log_level_name = "DEBUG"
        
        log_level = getattr(logging, log_level_name)
        log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        log_file = log_config.get("file")
        
        # 创建日志目录
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 配置根日志记录器
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8') if log_file else logging.NullHandler(),
                logging.StreamHandler()
            ]
        )
        
        # 获取日志记录器
        self.logger = logging.getLogger("chat_cli")
    
    def _create_data_dirs(self):
        """创建数据目录"""
        # 会话数据目录
        sessions_dir = os.path.join(
            self.config_manager.get_value("storage.base_path", "./data"),
            "sessions"
        )
        os.makedirs(sessions_dir, exist_ok=True)
        self.sessions_dir = sessions_dir
        
        # 当前会话目录
        session_dir = os.path.join(sessions_dir, self.session_id)
        os.makedirs(session_dir, exist_ok=True)
        self.session_dir = session_dir
    
    def _create_mock_session(self) -> ChatSession:
        """
        创建模拟聊天会话
        
        在实际应用中，应使用真实的RAG管道和LLM
        但由于这些组件在此原型中尚未完全实现，使用模拟会话

        Returns:
            ChatSession: 模拟的聊天会话
        """
        # 创建会话
        session = ChatSession(session_id=self.session_id)
        
        # 添加系统消息
        session.add_message(
            role="system",
            content="这是一个基于学术文献内容的智能问答助手。您可以询问有关已上传文献的问题，系统将尝试根据文献内容提供回答。"
        )
        
        self.logger.info(f"已创建模拟聊天会话: {self.session_id}")
        return session
    
    def _parse_args(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(
            description="聊天对话命令行界面 - 提供基于文档内容的自然语言对话",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例用法:
  # 启动新的聊天会话
  python chat_cli.py
  
  # 继续特定会话
  python chat_cli.py --session SESSION_ID
  
  # 使用特定的配置文件
  python chat_cli.py --config path/to/config.json
  
  # 回放历史聊天记录
  python chat_cli.py --session SESSION_ID --replay
            """
        )
        
        # 参数定义
        parser.add_argument("--session", help="会话ID，如不指定则创建新会话")
        parser.add_argument("--config", default=self.config_path, help="配置文件路径")
        parser.add_argument("--replay", action="store_true", help="回放历史聊天记录")
        parser.add_argument("--list", action="store_true", help="列出所有会话")
        parser.add_argument("--export", help="导出会话记录，需要指定会话ID")
        parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")
        
        return parser.parse_args()
    
    def run(self):
        """运行命令行界面"""
        args = self._parse_args()
        
        # 更新配置文件路径
        if args.config != self.config_path:
            self.config_path = args.config
            self.config_manager = ConfigManager(self.config_path)
        
        # 更新日志级别
        if args.verbose:
            self.verbose = True
            logging.getLogger().setLevel(logging.DEBUG)
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug("已启用详细日志模式")
        
        # 处理命令
        if args.list:
            self._list_sessions()
        elif args.export:
            self._export_session(args.export)
        else:
            # 正常聊天模式
            if args.session:
                self.session_id = args.session
                # 加载现有会话
                self._load_session(args.session)
            
            # 如果是回放模式
            if args.replay:
                self._replay_session()
            else:
                # 开始交互式聊天
                self._start_interactive_chat()
    
    def _list_sessions(self):
        """列出所有会话"""
        sessions = []
        
        try:
            # 遍历会话目录
            for session_id in os.listdir(self.sessions_dir):
                session_dir = os.path.join(self.sessions_dir, session_id)
                session_file = os.path.join(session_dir, "session.json")
                
                if os.path.isdir(session_dir) and os.path.exists(session_file):
                    try:
                        with open(session_file, 'r', encoding='utf-8') as f:
                            session_data = json.load(f)
                        
                        # 提取基本信息
                        sessions.append({
                            "session_id": session_id,
                            "created_at": session_data.get("created_at", "未知"),
                            "last_active_at": session_data.get("last_active_at", "未知"),
                            "message_count": len(session_data.get("messages", [])),
                            "metadata": session_data.get("metadata", {})
                        })
                    except Exception as e:
                        self.logger.warning(f"读取会话 {session_id} 信息时出错: {str(e)}")
            
            # 按最后活动时间排序
            sessions.sort(key=lambda x: x.get("last_active_at", ""), reverse=True)
            
            # 输出会话列表
            if not sessions:
                print("没有找到会话记录")
                return
            
            # 打印表头
            print(f"{'会话ID':<36} | {'创建时间':<20} | {'最后活动时间':<20} | {'消息数':<8}")
            print("-" * 90)
            
            # 打印每个会话
            for session in sessions:
                created_at = self._format_timestamp(session.get("created_at", ""))
                last_active_at = self._format_timestamp(session.get("last_active_at", ""))
                print(f"{session.get('session_id', 'N/A'):<36} | {created_at:<20} | {last_active_at:<20} | {session.get('message_count', 0):<8}")
            
            self.logger.info(f"列出了 {len(sessions)} 个会话")
            
        except Exception as e:
            self.logger.error(f"列出会话时出错: {str(e)}")
            print(f"错误：列出会话时出错：{str(e)}")
    
    def _export_session(self, session_id: str):
        """导出会话记录"""
        session_dir = os.path.join(self.sessions_dir, session_id)
        session_file = os.path.join(session_dir, "session.json")
        
        if not os.path.exists(session_file):
            self.logger.error(f"会话不存在：{session_id}")
            print(f"错误：会话ID {session_id} 不存在")
            return
        
        try:
            # 读取会话数据
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # 创建输出文件名
            output_file = f"session_{session_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
            
            # 导出为文本格式
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"会话ID: {session_id}\n")
                f.write(f"创建时间: {self._format_timestamp(session_data.get('created_at', ''))}\n")
                f.write(f"最后活动时间: {self._format_timestamp(session_data.get('last_active_at', ''))}\n\n")
                
                f.write("聊天记录:\n")
                f.write("=" * 80 + "\n\n")
                
                for msg in session_data.get("messages", []):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    timestamp = self._format_timestamp(msg.get("timestamp", ""))
                    
                    if role == "system":
                        continue  # 跳过系统消息
                    
                    f.write(f"[{timestamp}] {role.upper()}:\n")
                    f.write(f"{content}\n\n")
                    
                    # 添加引用信息（如果有）
                    if "citations" in msg:
                        f.write("引用来源:\n")
                        for citation in msg.get("citations", []):
                            doc_id = citation.get("document_id", "未知")
                            text = citation.get("text", "")
                            f.write(f"- 文档 {doc_id}: {text}\n")
                        f.write("\n")
                    
                    f.write("-" * 80 + "\n\n")
            
            print(f"会话记录已导出至: {output_file}")
            self.logger.info(f"已导出会话 {session_id} 的记录到 {output_file}")
            
        except Exception as e:
            self.logger.error(f"导出会话 {session_id} 时出错: {str(e)}")
            print(f"错误：导出会话时出错：{str(e)}")
    
    def _load_session(self, session_id: str):
        """加载现有会话"""
        session_dir = os.path.join(self.sessions_dir, session_id)
        session_file = os.path.join(session_dir, "session.json")
        
        if not os.path.exists(session_file):
            self.logger.warning(f"会话 {session_id} 不存在，将创建新会话")
            print(f"会话ID {session_id} 不存在，已创建新会话")
            # 更新会话目录
            self.session_dir = session_dir
            os.makedirs(self.session_dir, exist_ok=True)
            return
        
        try:
            # 读取会话数据
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # 创建会话对象
            self.session = ChatSession.from_dict(session_data)
            
            # 更新会话目录
            self.session_dir = session_dir
            
            self.logger.info(f"已加载会话: {session_id}")
            print(f"已加载会话ID: {session_id}")
            
        except Exception as e:
            self.logger.error(f"加载会话 {session_id} 时出错: {str(e)}")
            print(f"错误：加载会话时出错：{str(e)}")
    
    def _save_session(self):
        """保存当前会话"""
        try:
            # 保存会话数据
            session_file = os.path.join(self.session_dir, "session.json")
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(self.session.to_dict(), f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"已保存会话: {self.session_id}")
            
        except Exception as e:
            self.logger.error(f"保存会话 {self.session_id} 时出错: {str(e)}")
    
    def _replay_session(self):
        """回放会话记录"""
        messages = self.session.get_messages()
        
        if not messages:
            print("会话中没有消息记录")
            return
        
        print("\n" + "=" * 80)
        print(f"回放会话 {self.session_id} 的历史记录")
        print("=" * 80 + "\n")
        
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if role == "system":
                continue  # 跳过系统消息
            
            timestamp = self._format_timestamp(msg.get("timestamp", ""))
            
            print(f"[{timestamp}] {role.upper()}:")
            print(f"{content}")
            
            # 添加引用信息（如果有）
            if "citations" in msg and msg["citations"]:
                print("\n引用来源:")
                for citation in msg.get("citations", []):
                    doc_id = citation.get("document_id", "未知")
                    text = citation.get("text", "")
                    print(f"- 文档 {doc_id}: {text}")
            
            print("\n" + "-" * 80 + "\n")
            
            # 为了模拟对话效果，增加短暂延迟
            time.sleep(0.5)
    
    def _start_interactive_chat(self):
        """开始交互式聊天"""
        print("\n" + "=" * 80)
        print("欢迎使用学术文献智能问答系统")
        print("输入问题与系统对话，输入'exit'或'quit'退出对话")
        print("=" * 80 + "\n")
        
        # 显示系统提示
        system_message = "我是一个基于学术文献内容的智能问答助手。您可以询问有关已上传文献的问题，我将尝试根据文献内容提供回答。"
        print(f"系统: {system_message}\n")
        
        # 聊天循环
        while True:
            # 获取用户输入
            try:
                user_input = input("您: ")
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n再见！")
                break
            
            # 检查退出命令
            if user_input.lower() in ["exit", "quit", "bye", "再见"]:
                print("\n再见！")
                break
            
            # 检查帮助命令
            if user_input.lower() in ["help", "?", "帮助"]:
                self._print_help()
                continue
            
            # 检查清屏命令
            if user_input.lower() in ["clear", "cls", "清屏"]:
                os.system("cls" if os.name == "nt" else "clear")
                continue
            
            # 检查保存命令
            if user_input.lower() in ["save", "保存"]:
                self._save_session()
                print("会话已保存")
                continue
            
            # 忽略空输入
            if not user_input.strip():
                continue
            
            # 处理用户输入
            self._process_user_input(user_input)
    
    def _process_user_input(self, user_input: str):
        """处理用户输入"""
        try:
            # 添加用户消息
            self.session.add_message(role="user", content=user_input)
            
            # 在实际应用中，应调用RAG管道处理查询
            # 由于RAG管道尚未实现，这里使用模拟回答
            print("\n助手: ", end="", flush=True)
            
            # 生成模拟回答
            response = self._generate_mock_response(user_input)
            
            # 模拟打字效果
            for char in response:
                print(char, end="", flush=True)
                time.sleep(0.01)  # 调整速度
            print("\n")
            
            # 添加助手消息
            self.session.add_message(role="assistant", content=response)
            
            # 保存会话
            self._save_session()
            
        except Exception as e:
            self.logger.error(f"处理用户输入时出错: {str(e)}")
            print(f"\n处理您的问题时出错: {str(e)}")
    
    def _generate_mock_response(self, query: str) -> str:
        """
        生成模拟回答
        
        在实际应用中，应使用RAG管道生成真实回答
        
        Args:
            query: 用户查询
            
        Returns:
            str: 模拟的回答
        """
        # 简单的模拟回答逻辑
        responses = {
            "帮助": "您可以询问关于已上传文献的任何问题，我将根据文献内容回答。例如，您可以问特定论文的主要观点、方法论、结论等。",
            "文档": "目前系统中已上传了多篇学术文献，包括关于人工智能、机器学习、自然语言处理等领域的研究论文。",
            "功能": "我可以回答关于文献内容的问题，摘要总结，提取关键观点，解释复杂概念，比较不同文献的观点等。",
            "检索": "我使用向量数据库技术进行相似度检索，找到与您问题最相关的文档片段，然后基于这些内容生成回答。"
        }
        
        # 检查是否有匹配的关键词
        for keyword, response in responses.items():
            if keyword in query:
                return response
        
        # 默认回答
        default_responses = [
            "根据文献内容，这个问题涉及到多个研究领域。相关研究表明，该领域仍有许多未解决的挑战和机会。",
            "基于检索到的文献，该问题的答案是多方面的。一方面，研究显示特定方法的有效性；另一方面，也存在一些局限性需要考虑。",
            "文献中对这个问题有不同观点。一些研究者认为A方法更有效，而另一些则支持B方法，具体取决于应用场景和条件。",
            "据分析的文献显示，这是一个活跃的研究方向，近年来有多项创新成果。最新的研究趋势表明技术正朝着更高效、更智能的方向发展。",
            "这个问题在现有文献中有详细讨论。根据X等人的研究，该方法在特定条件下表现优异；而Y等人的研究则提出了一些改进方案。"
        ]
        
        import random
        return random.choice(default_responses)
    
    def _print_help(self):
        """打印帮助信息"""
        print("\n" + "=" * 80)
        print("帮助信息")
        print("=" * 80)
        print("- 输入问题与系统对话")
        print("- 输入'exit'或'quit'退出对话")
        print("- 输入'help'或'?'显示帮助信息")
        print("- 输入'clear'或'cls'清屏")
        print("- 输入'save'保存当前会话")
        print("=" * 80 + "\n")
    
    def _format_timestamp(self, timestamp) -> str:
        """格式化时间戳"""
        if not timestamp:
            return "未知"
        
        try:
            dt = datetime.fromtimestamp(float(timestamp))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return str(timestamp)


def main():
    """主入口函数"""
    try:
        cli = ChatCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n操作已取消")
        sys.exit(1)
    except Exception as e:
        print(f"错误：{str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
