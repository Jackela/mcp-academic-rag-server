#!/usr/bin/env python3
"""
Standalone MCP Academic RAG Server - 渐进式增强版本
重用现有架构但提供独立的依赖管理
"""

import asyncio
import json
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid
import time

# 配置日志到stderr (MCP要求) - 必须在其他模块引用logger之前
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("mcp-academic-rag-standalone")

# 重用现有模型和核心组件
try:
    from models.document import Document
    from core.server_context import ServerContext
    from rag.haystack_pipeline import RAGPipeline
    from connectors.haystack_llm_connector import HaystackLLMConnector
    from document_stores.haystack_store import HaystackDocumentStore
    from processors.haystack_embedding_processor import HaystackEmbeddingProcessor
    HAS_FULL_DEPS = True
    logger.info("✅ 完整RAG依赖已加载")
except ImportError as e:
    # 如果依赖不可用，使用简化版本
    HAS_FULL_DEPS = False
    logger.warning(f"⚠️ 完整RAG依赖不可用: {e}")
    
    class Document:
        """简化的文档类"""
        def __init__(self, file_path: str):
            self.document_id = str(uuid.uuid4())
            self.file_path = file_path
            self.file_name = os.path.basename(file_path)
            self.file_type = os.path.splitext(file_path)[1].lower()
            self.creation_time = time.strftime("%Y-%m-%d %H:%M:%S")
            self.status = "new"
            self.metadata = {}
            self.content = {}
        
        def store_content(self, stage: str, content: str):
            """存储内容到指定阶段"""
            self.content[stage] = content
        
        def get_content(self, stage: str):
            """获取指定阶段的内容"""
            return self.content.get(stage)
    
    class ServerContext:
        """简化的服务器上下文"""
        def __init__(self):
            self._initialized = False
        
        def initialize(self):
            self._initialized = True
        
        @property
        def is_initialized(self):
            return self._initialized
        
        def get_status(self):
            return {
                "initialized": self._initialized,
                "config_loaded": False,
                "pipeline_ready": False,
                "rag_enabled": False,
                "processors_count": 0
            }

class SimpleDocumentProcessor:
    """增强的文档处理器 - 支持PDF并重用Document模型"""
    
    def __init__(self, mcp_server=None):
        self.documents: Dict[str, Document] = {}
        self.mcp_server = mcp_server  # 引用父服务器以访问RAG组件
    
    def process_pdf(self, file_path: str) -> str:
        """处理PDF文件 - 尝试多种方法"""
        try:
            # 方法1: 尝试使用PyPDF2 (如果可用)
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    if text.strip():
                        return text
            except ImportError:
                pass
            except Exception:
                pass
            
            # 方法2: 尝试使用pdfplumber (如果可用)
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    if text.strip():
                        return text
            except ImportError:
                pass
            except Exception:
                pass
            
            # 方法3: 尝试使用pdfminer (如果可用)
            try:
                from pdfminer.high_level import extract_text
                text = extract_text(file_path)
                if text.strip():
                    return text
            except ImportError:
                pass
            except Exception:
                pass
            
            # 方法4: 使用现有的OCR处理器 (如果可用)
            if HAS_FULL_DEPS:
                try:
                    from processors.ocr_processor import OCRProcessor
                    processor = OCRProcessor()
                    # 这里需要先将PDF转换为图像，然后OCR
                    # 为简化，我们暂时跳过这个方法
                    pass
                except ImportError:
                    pass
            
            # 如果所有方法都失败，返回错误信息
            raise Exception("PDF处理失败：需要安装 PyPDF2, pdfplumber 或 pdfminer3 之一")
            
        except Exception as e:
            raise Exception(f"PDF处理错误: {str(e)}")
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """处理文档并存储 - 支持PDF和完整RAG管道"""
        try:
            document = Document(file_path)
            file_ext = document.file_type.lower()
            
            # 读取文件内容
            if file_ext == '.pdf':
                content = self.process_pdf(file_path)
                document.metadata['processing_method'] = 'pdf_extraction'
            else:
                # 处理文本文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                document.metadata['processing_method'] = 'text_file'
            
            # 存储原始内容
            document.content['raw_text'] = content
            document.metadata.update({
                'word_count': len(content.split()),
                'char_count': len(content),
                'line_count': len(content.splitlines())
            })
            
            # 如果完整RAG可用，使用嵌入处理器
            embedding_result = None
            if (self.mcp_server and 
                hasattr(self.mcp_server, 'full_rag_available') and 
                self.mcp_server.full_rag_available and 
                hasattr(self.mcp_server, 'embedding_processor')):
                try:
                    logger.info(f"🚀 使用完整RAG管道处理文档: {document.file_name}")
                    
                    # 使用嵌入处理器处理文档
                    # 首先为文档添加OCR处理的内容（模拟已处理）
                    document.store_content("OCRProcessor", content)
                    
                    # 使用嵌入处理器生成向量嵌入
                    embedding_result = self.mcp_server.embedding_processor.process(document)
                    
                    if embedding_result.is_successful():
                        result_data = embedding_result.get_data()
                        document.metadata.update({
                            'chunks_count': result_data.get('chunks', 0),
                            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                            'rag_processing': 'completed'
                        })
                        logger.info(f"✅ 文档嵌入处理完成: {result_data.get('chunks', 0)} 个块")
                    else:
                        logger.warning(f"⚠️ 嵌入处理失败: {embedding_result.get_message()}")
                        document.metadata['rag_processing'] = 'failed'
                        
                except Exception as e:
                    logger.error(f"❌ 嵌入处理异常: {e}", exc_info=True)
                    document.metadata['rag_processing'] = 'error'
            else:
                logger.info(f"🔍 使用简单关键词匹配模式处理文档: {document.file_name}")
                document.metadata['rag_processing'] = 'simple_mode'
            
            document.status = "completed"
            
            # 存储文档
            self.documents[document.document_id] = document
            
            return {
                "success": True,
                "document_id": document.document_id,
                "document": document,
                "embedding_result": embedding_result.get_data() if embedding_result and embedding_result.is_successful() else None
            }
            
        except Exception as e:
            logger.error(f"❌ 文档处理异常: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """搜索文档 - 简单关键词匹配"""
        results = []
        query_lower = query.lower()
        
        for doc_id, doc in self.documents.items():
            content = doc.content.get('raw_text', '')
            content_lower = content.lower()
            
            # 计算匹配分数
            score = 0
            for word in query_lower.split():
                score += content_lower.count(word)
            
            if score > 0:
                # 提取相关片段
                snippet_start = max(0, content_lower.find(query_lower.split()[0]) - 50)
                snippet_end = min(len(content), snippet_start + 200)
                snippet = content[snippet_start:snippet_end]
                
                results.append({
                    "document": doc,
                    "score": score,
                    "snippet": snippet.strip()
                })
        
        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

class MCPServer:
    """增强的MCP服务器实现 - 重用现有架构"""
    
    def __init__(self):
        # 使用依赖注入模式
        self.server_context = ServerContext()
        self.document_processor = SimpleDocumentProcessor(mcp_server=self)
        
        # 尝试初始化真正的RAG管道
        self._try_initialize_full_rag()
        
        # 扩展工具列表 - 重用SDK架构
        self.tools = [
            {
                "name": "test_connection",
                "description": "测试MCP服务器连接状态",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "测试消息",
                            "default": "Hello MCP!"
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "validate_system", 
                "description": "验证服务器系统配置和依赖",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "process_document",
                "description": "处理学术文档并建立索引（支持文本文件）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "要处理的文档文件路径"
                        },
                        "file_name": {
                            "type": "string",
                            "description": "可选的文档名称"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "query_documents",
                "description": "使用RAG查询已处理的文档",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "要查询的问题"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "可选的会话ID用于上下文"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "返回的相关文档数量",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "list_documents",
                "description": "列出所有已处理的文档",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]
    
    def validate_api_key(self, api_key: str) -> bool:
        """验证OpenAI API密钥格式"""
        if not api_key or not isinstance(api_key, str):
            return False
        if api_key != api_key.strip() or ' ' in api_key:
            return False
        if not api_key.startswith('sk-'):
            return False
        if len(api_key) < 21:
            return False
        return True
    
    def validate_environment(self) -> Dict[str, Any]:
        """验证环境配置"""
        result = {
            "api_key_valid": False,
            "data_path_exists": False,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "working_directory": os.getcwd(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 检查API密钥
        api_key = os.environ.get('OPENAI_API_KEY')
        result["api_key_valid"] = self.validate_api_key(api_key)
        result["api_key_length"] = len(api_key) if api_key else 0
        
        # 检查数据目录
        data_path = os.environ.get('DATA_PATH', './data')
        try:
            Path(data_path).mkdir(parents=True, exist_ok=True)
            result["data_path_exists"] = True
            result["data_path"] = str(Path(data_path).absolute())
        except Exception as e:
            result["data_path_error"] = str(e)
        
        return result
    
    async def handle_list_tools(self, request_id: Any) -> Dict[str, Any]:
        """处理工具列表请求"""
        logger.info("Handling tools/list request")
        return {
            "jsonrpc": "2.0",
            "result": {
                "tools": self.tools
            },
            "id": request_id
        }
    
    async def handle_call_tool(self, request_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理工具调用请求"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        logger.info(f"Handling tool call: {tool_name} with args: {arguments}")
        
        try:
            if tool_name == "test_connection":
                return await self.handle_test_connection(request_id, arguments)
                
                
            elif tool_name == "validate_system":
                return await self.handle_validate_system(request_id, arguments)
                
            elif tool_name == "process_document":
                return await self.handle_process_document(request_id, arguments)
                
            elif tool_name == "query_documents":
                return await self.handle_query_documents(request_id, arguments)
                
            elif tool_name == "list_documents":
                return await self.handle_list_documents(request_id, arguments)
                
            else:
                result_text = f"❌ 未知工具: {tool_name}\n可用工具: {', '.join([t['name'] for t in self.tools])}"
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": result_text
                            }
                        ]
                    },
                    "id": request_id
                }
            
        except Exception as e:
            logger.error(f"Tool call error: {str(e)}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"工具执行错误: {str(e)}"
                },
                "id": request_id
            }
    
    async def handle_initialize(self, request_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理初始化请求"""
        logger.info("Handling initialize request")
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {
                    "tools": {
                        "listChanged": False
                    }
                },
                "serverInfo": {
                    "name": "academic-rag-server-standalone",
                    "version": "1.0.0-standalone"
                }
            },
            "id": request_id
        }
    
    def _try_initialize_full_rag(self):
        """尝试初始化完整的RAG管道"""
        try:
            logger.info("🔄 尝试初始化完整RAG管道...")
            
            # 检查是否有API密钥
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key or not self.validate_api_key(api_key):
                logger.warning("❌ API密钥无效，使用简单检索模式")
                self.full_rag_available = False
                return
            
            # 尝试初始化完整RAG系统
            if HAS_FULL_DEPS:
                try:
                    # 初始化文档存储 (InMemoryDocumentStore with embedding support)
                    self.document_store = HaystackDocumentStore()
                    logger.info("✅ 文档存储已初始化")
                    
                    # 初始化嵌入处理器
                    self.embedding_processor = HaystackEmbeddingProcessor(
                        document_store=self.document_store,
                        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    logger.info("✅ 嵌入处理器已初始化")
                    
                    # 初始化LLM连接器
                    self.llm_connector = HaystackLLMConnector(
                        api_key=api_key,
                        model="gpt-3.5-turbo",
                        parameters={"temperature": 0.7}
                    )
                    logger.info("✅ LLM连接器已初始化")
                    
                    # 初始化RAG管道
                    self.rag_pipeline = RAGPipeline(
                        llm_connector=self.llm_connector,
                        document_store=self.document_store.get_haystack_store(),
                        retriever_top_k=5
                    )
                    logger.info("✅ RAG管道已初始化")
                    
                    self.full_rag_available = True
                    logger.info("🎉 完整RAG管道初始化成功 (Sentence-BERT + OpenAI)")
                    return
                    
                except Exception as e:
                    logger.error(f"❌ RAG管道初始化失败: {e}", exc_info=True)
            
            self.full_rag_available = False
            logger.info("⚠️ 使用简单检索模式")
            
        except Exception as e:
            logger.error(f"❌ RAG初始化错误: {e}", exc_info=True)
            self.full_rag_available = False
    
    async def handle_test_connection(self, request_id: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """处理连接测试 - 重用SDK模式"""
        message = arguments.get("message", "Hello MCP!")
        
        result_text = f"""✅ MCP Academic RAG Server (Enhanced) 连接成功!
📩 收到消息: {message}
🕐 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
🐍 Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}
📁 工作目录: {os.getcwd()}
📚 已处理文档: {len(self.document_processor.documents)}
🔧 依赖状态: {'完整依赖' if HAS_FULL_DEPS else '简化模式'}
🛠️ 服务器上下文: {'已初始化' if self.server_context.is_initialized else '未初始化'}
🧠 RAG管道: {'✅ 可用 (Sentence-BERT + OpenAI)' if hasattr(self, 'full_rag_available') and self.full_rag_available else '❌ 不可用 (关键词匹配)'}
🎯 嵌入模型: {'sentence-transformers/all-MiniLM-L6-v2' if hasattr(self, 'full_rag_available') and self.full_rag_available else '无'}
🤖 生成模型: {'OpenAI GPT-3.5-turbo' if hasattr(self, 'full_rag_available') and self.full_rag_available else '无'}"""
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "content": [{
                    "type": "text",
                    "text": result_text
                }]
            },
            "id": request_id
        }
    
    async def handle_validate_system(self, request_id: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """处理系统验证 - 重用现有逻辑"""
        env_status = self.validate_environment()
        context_status = self.server_context.get_status()
        
        result_text = f"""🔍 系统验证报告:

🔑 API密钥: {'有效' if env_status['api_key_valid'] else '无效'} (长度: {env_status['api_key_length']})
📁 数据目录: {'存在' if env_status['data_path_exists'] else '不存在'}
🐍 Python版本: {env_status['python_version']}
📂 工作目录: {env_status['working_directory']}
🕐 检查时间: {env_status['timestamp']}

🏗️ 服务器状态:
  • 服务器上下文: {'已初始化' if context_status['initialized'] else '未初始化'}
  • 配置管理器: {'已加载' if context_status['config_loaded'] else '未加载'}
  • 文档管道: {'就绪' if context_status['pipeline_ready'] else '未就绪'}
  • RAG管道: {'启用' if context_status['rag_enabled'] else '禁用'}
  • 处理器数量: {context_status['processors_count']}

📚 文档处理状态:
  • 已处理文档: {len(self.document_processor.documents)}
  • 依赖模式: {'完整依赖' if HAS_FULL_DEPS else '简化模式'}"""
        
        if 'data_path' in env_status:
            result_text += f"\n📂 数据路径: {env_status['data_path']}"
        if 'data_path_error' in env_status:
            result_text += f"\n❌ 数据路径错误: {env_status['data_path_error']}"
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "content": [{
                    "type": "text",
                    "text": result_text
                }]
            },
            "id": request_id
        }
    
    async def handle_process_document(self, request_id: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """处理文档处理请求 - 重用Document模型"""
        file_path = arguments.get("file_path", "")
        file_name = arguments.get("file_name", "")
        
        if not file_path:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32602,
                    "message": "缺少必需参数: file_path"
                },
                "id": request_id
            }
        
        if not os.path.exists(file_path):
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [{
                        "type": "text",
                        "text": f"❌ 文件不存在: {file_path}"
                    }]
                },
                "id": request_id
            }
        
        # 检查支持的文件类型
        file_ext = Path(file_path).suffix.lower()
        text_types = ['.txt', '.md', '.rst', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv']
        pdf_types = ['.pdf']
        supported_types = text_types + pdf_types
        
        if file_ext not in supported_types:
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [{
                        "type": "text",
                        "text": f"❌ 暂不支持的文件类型: {file_ext}\n支持的类型:\n• 文本文件: {', '.join(text_types)}\n• PDF文件: {', '.join(pdf_types)}"
                    }]
                },
                "id": request_id
            }
        
        # 处理文档
        result = self.document_processor.process_document(file_path)
        
        if result["success"]:
            doc = result["document"]
            embedding_result = result.get("embedding_result")
            
            result_text = f"""✅ 文档处理成功!

📄 文件信息:
  • 路径: {file_path}
  • 名称: {file_name if file_name else doc.file_name}
  • 类型: {file_ext}
  • 状态: {doc.status}

🆔 文档ID: {doc.document_id}

📊 内容统计:
  • 字符数: {doc.metadata.get('char_count', 0):,}
  • 单词数: {doc.metadata.get('word_count', 0):,}
  • 行数: {doc.metadata.get('line_count', 0):,}

🧠 RAG处理状态: {doc.metadata.get('rag_processing', '未知')}"""

            # 如果有嵌入处理结果，显示详细信息
            if embedding_result:
                result_text += f"""
📊 向量嵌入信息:
  • 文档分块数: {embedding_result.get('chunks', 0)}
  • 嵌入模型: sentence-transformers/all-MiniLM-L6-v2
  • 向量维度: 384维"""

            result_text += f"""

✅ 文档已成功处理并建立索引，可以使用 query_documents 工具进行{'智能RAG查询' if doc.metadata.get('rag_processing') == 'completed' else '关键词查询'}。"""
        else:
            result_text = f"""❌ 文档处理失败:

📄 文件路径: {file_path}
💬 错误信息: {result['error']}

请检查:
• 文件是否可读
• 文件编码是否正确 (建议UTF-8)
• 文件权限是否充足"""
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "content": [{
                    "type": "text",
                    "text": result_text
                }]
            },
            "id": request_id
        }
    
    async def handle_query_documents(self, request_id: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """处理文档查询请求 - 完整RAG实现"""
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 5)
        session_id = arguments.get("session_id", "")
        
        if not query:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32602,
                    "message": "缺少必需参数: query"
                },
                "id": request_id
            }
        
        if len(self.document_processor.documents) == 0:
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [{
                        "type": "text",
                        "text": "❌ 没有已处理的文档。\n\n请先使用 process_document 工具处理一些文档。"
                    }]
                },
                "id": request_id
            }
        
        try:
            # 如果完整RAG可用，使用RAG管道
            if hasattr(self, 'full_rag_available') and self.full_rag_available and hasattr(self, 'rag_pipeline'):
                logger.info(f"🤖 使用完整RAG管道查询: {query}")
                
                try:
                    # 运行RAG管道
                    rag_result = self.rag_pipeline.run(
                        query=query,
                        chat_history=[],  # 简化实现，不使用历史记录
                        filters=None,  # 不使用过滤器
                        generation_kwargs={"temperature": 0.7}
                    )
                    
                    if 'error' not in rag_result:
                        # RAG成功，返回完整的智能答案
                        answer = rag_result.get("answer", "无法生成答案")
                        documents = rag_result.get("documents", [])
                        
                        result_text = f"""🤖 智能RAG查询结果:

❓ 问题: {query}

💬 AI答案:
{answer}

📁 相关文档片段 (前{len(documents[:top_k])}个):
"""
                        
                        for i, doc in enumerate(documents[:top_k], 1):
                            content = doc.get("content", "")
                            if len(content) > 200:
                                content = content[:200] + "..."
                            
                            metadata = doc.get("metadata", {})
                            file_name = metadata.get("file_name", "未知")
                            
                            result_text += f"\n{i}. 📄 {file_name}\n   📝 {content}\n"
                        
                        result_text += f"\n🔍 检索模式: 嵌入向量匹配 (Sentence-BERT)"
                        result_text += f"\n🤖 生成模式: OpenAI GPT (LLM)"
                        
                        if session_id:
                            result_text += f"\n🆔 会话ID: {session_id}"
                        
                        return {
                            "jsonrpc": "2.0",
                            "result": {
                                "content": [{
                                    "type": "text",
                                    "text": result_text
                                }]
                            },
                            "id": request_id
                        }
                    else:
                        logger.warning(f"⚠️ RAG管道返回错误: {rag_result.get('error')}")
                        
                except Exception as e:
                    logger.error(f"❌ RAG管道执行异常: {e}", exc_info=True)
                    # 回退到简单模式
            
            # 回退到简单关键词匹配模式
            logger.info(f"🔍 使用简单关键词匹配模式查询: {query}")
            results = self.document_processor.search_documents(query, top_k)
            
            if not results:
                result_text = f"""🔍 查询结果:

❓ 查询: {query}
📊 结果: 未找到相关文档

💡 建议:
• 尝试更通用的关键词
• 检查拼写是否正确
• 确认相关内容已被处理

🔍 检索模式: 关键词匹配 (简单模式)"""
            else:
                result_text = f"""🔍 关键词匹配查询结果:

❓ 查询: {query}
📊 找到 {len(results)} 个相关文档:

"""
                for i, result in enumerate(results, 1):
                    doc = result["document"]
                    result_text += f"""📄 结果 {i}:
  • 文件: {doc.file_name}
  • 文档ID: {doc.document_id[:8]}...
  • 相关性: {result['score']} 分
  • 字数: {doc.metadata.get('word_count', 0):,}
  • 预览: {result['snippet'][:150]}...

"""
                result_text += f"\n🔍 检索模式: 关键词匹配 (简单模式)"
                
            if session_id:
                result_text += f"\n🆔 会话ID: {session_id}"
            
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [{
                        "type": "text",
                        "text": result_text
                    }]
                },
                "id": request_id
            }
            
        except Exception as e:
            logger.error(f"❌ 查询执行异常: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"查询执行失败: {str(e)}"
                },
                "id": request_id
            }
    
    async def handle_list_documents(self, request_id: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """处理文档列表请求"""
        docs = self.document_processor.documents
        
        if not docs:
            result_text = """📚 文档列表:

暂无已处理的文档。

使用 process_document 工具来处理文档:
• 支持的格式: .txt, .md, .rst, .py, .js, .html, .css, .json, .xml, .csv
• 处理后可使用 query_documents 进行查询"""
        else:
            result_text = f"📚 文档列表 (共 {len(docs)} 个):\n\n"
            
            for i, (doc_id, doc) in enumerate(docs.items(), 1):
                result_text += f"""{i}. 📄 {doc.file_name}
   🆔 ID: {doc_id[:8]}...
   📊 {doc.metadata.get('word_count', 0):,} 词, {doc.metadata.get('char_count', 0):,} 字符
   📁 路径: {doc.file_path}
   ⏰ 创建: {doc.creation_time}
   🔄 状态: {doc.status}

"""
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "content": [{
                    "type": "text",
                    "text": result_text
                }]
            },
            "id": request_id
        }
    
    async def run(self):
        """运行MCP服务器"""
        logger.info("Starting MCP Academic RAG Server (Standalone)")
        
        try:
            while True:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line.strip():
                    continue
                
                try:
                    request = json.loads(line)
                    method = request.get("method")
                    request_id = request.get("id")
                    params = request.get("params", {})
                    
                    logger.debug(f"Received request: {method}")
                    
                    if method == "initialize":
                        response = await self.handle_initialize(request_id, params)
                    elif method == "tools/list":
                        response = await self.handle_list_tools(request_id)
                    elif method == "tools/call":
                        response = await self.handle_call_tool(request_id, params)
                    else:
                        response = {
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32601,
                                "message": f"Method not found: {method}"
                            },
                            "id": request_id
                        }
                    
                    print(json.dumps(response))
                    sys.stdout.flush()
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Request handling error: {e}", exc_info=True)
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}"
                        },
                        "id": request.get("id") if 'request' in locals() else None
                    }
                    print(json.dumps(error_response))
                    sys.stdout.flush()
                    
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        except Exception as e:
            logger.error(f"Server error: {str(e)}", exc_info=True)

def main():
    """主入口点"""
    if len(sys.argv) > 1 and sys.argv[1] == "--validate-only":
        # 验证模式
        server = MCPServer()
        env_status = server.validate_environment()
        
        if env_status["api_key_valid"]:
            print("✅ 独立服务器环境验证通过", file=sys.stderr)
            sys.exit(0)
        else:
            print("❌ API密钥验证失败", file=sys.stderr)
            sys.exit(1)
    else:
        # MCP服务器模式
        server = MCPServer()
        asyncio.run(server.run())

if __name__ == "__main__":
    main()