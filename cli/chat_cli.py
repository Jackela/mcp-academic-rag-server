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
from rag.haystack_pipeline import RAGPipeline, RAGPipelineFactory
from connectors.haystack_llm_connector import HaystackLLMConnector, HaystackLLMFactory


class ChatCLI:
    """聊天对话命令行界面类"""
    
    def __init__(self):
        """初始化聊天对话CLI"""
        try:
            # 设置日志
            self._setup_logging()
            
            # 生成唯一的会话ID
            self.session_id = str(uuid.uuid4())
            
            # 解析命令行参数
            self.args = self._parse_args()
            
            # 如果提供了会话ID参数，使用它
            if hasattr(self.args, 'session') and self.args.session:
                self.session_id = self.args.session
            
            # 创建数据目录
            self._create_data_dirs()
            
            # 记录初始化信息
            self.logger.info(f"聊天对话CLI初始化完成，会话ID：{self.session_id}")
            
            # 初始化组件
            self.config_manager = ConfigManager()
            self.session_manager = ChatSessionManager()
            self.rag_pipeline = None
            self.session = None
            
            # 初始化RAG管道
            self._initialize_rag_pipeline()
            
            # 初始化聊天会话
            self.session = self._create_session()
            
        except Exception as e:
            print(f"初始化聊天对话CLI失败: {str(e)}")
            sys.exit(1)
    
    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('chat_cli.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('ChatCLI')
    
    def _create_data_dirs(self):
        """创建必要的数据目录"""
        # 创建会话目录
        base_dir = os.path.join(os.getcwd(), "data", "sessions")
        self.sessions_dir = base_dir
        
        # 当前会话目录
        session_dir = os.path.join(base_dir, self.session_id)
        os.makedirs(session_dir, exist_ok=True)
        self.session_dir = session_dir
    
    def _initialize_rag_pipeline(self):
        """
        初始化RAG管道
        """
        try:
            # 从配置中获取LLM设置
            llm_config = self.config_manager.get_value("llm", {})
            
            # 创建LLM连接器
            llm_connector = HaystackLLMConnector(config=llm_config)
            
            # 创建RAG管道
            rag_config = self.config_manager.get_value("rag_settings", {})
            self.rag_pipeline = RAGPipelineFactory.create_pipeline(
                llm_connector=llm_connector,
                config=rag_config
            )
            
            self.logger.info("成功初始化RAG管道")
            
        except Exception as e:
            self.logger.error(f"初始化RAG管道失败: {str(e)}")
            self.rag_pipeline = None
    
    def _create_session(self) -> ChatSession:
        """
        创建聊天会话（使用真实的RAG管道）
        
        Returns:
            ChatSession: 聊天会话
        """
        # 使用会话管理器创建会话
        session = self.session_manager.create_session(
            session_id=self.session_id,
            metadata={"cli_session": True, "created_by": "chat_cli"}
        )
        
        # 设置RAG管道
        if self.rag_pipeline:
            session.set_rag_pipeline(self.rag_pipeline)
        
        self.logger.info(f"已创建聊天会话: {self.session_id}")
        return session
    
    def _parse_args(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(
            description="聊天对话命令行界面 - 提供基于文档内容的自然语言对话",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用示例:
  python chat_cli.py                    # 开始新的聊天会话
  python chat_cli.py --session abc123   # 继续指定的会话
  python chat_cli.py --list             # 列出所有会话
  python chat_cli.py --export abc123    # 导出会话记录
  python chat_cli.py --replay abc123    # 回放会话记录

注意:
  - 会话记录会自动保存
  - 使用 Ctrl+C 或输入 "exit" 退出
  - 输入 "help" 查看帮助信息
            """
        )
        
        # 互斥参数组 - 只能选择一个操作
        action_group = parser.add_mutually_exclusive_group()
        
        action_group.add_argument(
            '--session', '-s',
            type=str,
            help='指定要继续的会话ID'
        )
        
        action_group.add_argument(
            '--list', '-l',
            action='store_true',
            help='列出所有会话记录'
        )
        
        action_group.add_argument(
            '--export', '-e',
            type=str, 
            metavar='SESSION_ID',
            help='导出指定会话的聊天记录'
        )
        
        action_group.add_argument(
            '--replay', '-r',
            type=str,
            metavar='SESSION_ID',
            help='回放指定会话的聊天记录'
        )
        
        # 其他参数
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='启用详细日志输出'
        )
        
        return parser.parse_args()
    
    def run(self):
        """运行聊天对话CLI"""
        try:
            # 根据参数执行相应操作
            if self.args.list:
                self._list_sessions()
            elif self.args.export:
                self._export_session(self.args.export)
            elif self.args.session:
                self.session_id = self.args.session
                # 加载现有会话
                self._load_session(self.args.session)
            
            # 如果是回放模式
            if self.args.replay:
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
                    if msg.get("message_id") in session_data.get("citations", {}):
                        citations = session_data["citations"][msg["message_id"]]
                        if citations:
                            f.write("参考文献:\n")
                            for citation in citations:
                                f.write(f"- {citation.get('text', '')}\n")
                            f.write("\n")
            
            print(f"会话记录已导出到: {output_file}")
            self.logger.info(f"会话 {session_id} 已导出到 {output_file}")
            
        except Exception as e:
            self.logger.error(f"导出会话失败: {str(e)}")
            print(f"错误：导出会话失败：{str(e)}")
    
    def _load_session(self, session_id: str):
        """加载现有会话"""
        # 使用会话管理器加载会话
        try:
            self.session = self.session_manager.get_session(session_id)
            if self.session:
                # 设置RAG管道
                if self.rag_pipeline:
                    self.session.set_rag_pipeline(self.rag_pipeline)
                self.logger.info(f"已加载会话: {session_id}")
            else:
                print(f"错误：会话ID {session_id} 不存在")
                return
        except Exception as e:
            self.logger.error(f"加载会话失败: {str(e)}")
            print(f"错误：加载会话失败：{str(e)}")
            return
    
    def _replay_session(self):
        """回放会话记录"""
        if not self.session:
            print("错误：没有可回放的会话")
            return
        
        print(f"\n开始回放会话 {self.session_id}")
        print("=" * 60)
        
        # 回放所有消息
        for msg in self.session.messages:
            if msg.role == "system":
                continue  # 跳过系统消息
            
            timestamp = self._format_timestamp(msg.timestamp)
            print(f"\n[{timestamp}] {msg.role.upper()}:")
            print(msg.content)
            
            # 如果有引用，显示引用信息
            citations = self.session.get_citations(msg.message_id)
            if citations:
                print("\n参考文献:")
                for citation in citations:
                    print(f"- {citation.text}")
            
            # 延迟一下，模拟真实对话节奏
            time.sleep(0.5)
        
        print("\n" + "=" * 60)
        print("会话回放完成")
    
    def _start_interactive_chat(self):
        """开始交互式聊天"""
        print("\n欢迎使用学术文献智能问答系统！")
        print("=" * 50)
        print(f"会话ID: {self.session_id}")
        print("输入您的问题，系统将基于已上传的文献为您回答。")
        print("命令: 'help' 查看帮助, 'exit' 退出, 'save' 保存会话")
        print("=" * 50)
        
        try:
            while True:
                # 获取用户输入
                try:
                    user_input = input("\n您: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n再见！")
                    break
                
                # 处理特殊命令
                if user_input.lower() in ['exit', 'quit', '退出']:
                    print("再见！")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'save':
                    self._save_session()
                    continue
                elif not user_input:
                    continue
                
                # 处理用户问题
                self._process_user_input(user_input)
                
        except Exception as e:
            self.logger.error(f"交互式聊天出错: {str(e)}")
            print(f"\n发生错误: {str(e)}")
        finally:
            # 保存会话
            self._save_session()
    
    def _process_user_input(self, user_input: str):
        """处理用户输入并生成回答"""
        try:
            # 使用RAG管道生成回答
            try:
                if self.rag_pipeline:
                    response = self.session.query(user_input)
                    answer = response.get("answer", "无法生成回答")
                    documents = response.get("documents", [])
                    
                    # 显示回答
                    print(f"\n助手: {answer}")
                    
                    # 显示参考文献（如果有）
                    if documents:
                        print("\n参考文献:")
                        for i, doc in enumerate(documents[:3], 1):  # 显示前3个相关文档
                            print(f"{i}. {doc.get('metadata', {}).get('file_name', '未知文档')}")
                            print(f"   内容片段: {doc.get('content', '')[:100]}...")
                else:
                    response = self._generate_fallback_response(user_input)
                    print(f"\n助手: {response}")
            except Exception as e:
                self.logger.error(f"生成回答失败: {str(e)}")
                response = f"抱歉，处理您的问题时发生错误: {str(e)}"
                print(f"\n助手: {response}")
            
        except Exception as e:
            self.logger.error(f"处理用户输入时出错: {str(e)}")
            print(f"\n处理您的问题时出错: {str(e)}")
    
    def _generate_fallback_response(self, query: str) -> str:
        """
        生成备用回答（当RAG管道不可用时）
        
        Args:
            query: 用户查询
            
        Returns:
            str: 备用回答
        """
        return f"抱歉，RAG系统当前不可用。请确保已正确配置并上传了文档。您的问题是: {query}"
    
    def _show_help(self):
        """显示帮助信息"""
        help_text = """
可用命令:
  help    - 显示此帮助信息
  save    - 保存当前会话
  exit    - 退出程序
  
使用说明:
  1. 直接输入您的问题，系统会基于已上传的文献回答
  2. 系统会显示相关的文献引用信息
  3. 会话会自动保存，下次可以通过 --session 参数继续
  
示例问题:
  - "这篇论文的主要贡献是什么？"
  - "请总结关于机器学习的内容"
  - "有哪些研究方法被提到？"
        """
        print(help_text)
    
    def _save_session(self):
        """保存会话"""
        try:
            self.session_manager.save_sessions()
            print("会话已保存")
            self.logger.info(f"会话 {self.session_id} 已保存")
        except Exception as e:
            self.logger.error(f"保存会话失败: {str(e)}")
            print(f"保存会话失败: {str(e)}")
    
    def _format_timestamp(self, timestamp):
        """格式化时间戳"""
        try:
            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp)
                return dt.strftime('%Y-%m-%d %H:%M')
            elif isinstance(timestamp, str):
                # 尝试解析字符串时间戳
                try:
                    ts = float(timestamp)
                    dt = datetime.fromtimestamp(ts)
                    return dt.strftime('%Y-%m-%d %H:%M')
                except ValueError:
                    return timestamp[:19] if len(timestamp) >= 19 else timestamp
            else:
                return str(timestamp)
        except Exception:
            return str(timestamp)


def main():
    """主函数"""
    try:
        # 创建并运行CLI
        cli = ChatCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"程序出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()