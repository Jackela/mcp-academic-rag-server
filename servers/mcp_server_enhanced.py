#!/usr/bin/env python3
"""
Enhanced MCP Academic RAG Server - 渐进式功能实现
先用简单实现验证MCP协议，然后逐步添加RAG功能
"""

import asyncio
import json
import sys
import os
import logging
import time
import uuid
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess

# 配置日志到stderr (MCP要求)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("mcp-academic-rag-enhanced")

class SimpleDocumentProcessor:
    """简单的文档处理器 - 不依赖外部库"""
    
    def __init__(self):
        self.processed_documents = {}
    
    def process_text_file(self, file_path: str) -> Dict[str, Any]:
        """处理文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单的文档分析
            doc_id = str(uuid.uuid4())
            self.processed_documents[doc_id] = {
                "id": doc_id,
                "file_path": file_path,
                "content": content,
                "word_count": len(content.split()),
                "char_count": len(content),
                "processed_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return {
                "success": True,
                "document_id": doc_id,
                "word_count": len(content.split()),
                "char_count": len(content)
            }
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """简单的文档搜索 - 基于关键词匹配"""
        results = []
        query_lower = query.lower()
        
        for doc_id, doc in self.processed_documents.items():
            content_lower = doc["content"].lower()
            
            # 简单的相关性评分：计算查询词在文档中的出现次数
            score = 0
            for word in query_lower.split():
                score += content_lower.count(word)
            
            if score > 0:
                # 提取相关片段
                content = doc["content"]
                snippet_start = max(0, content_lower.find(query_lower.split()[0]) - 50)
                snippet_end = min(len(content), snippet_start + 200)
                snippet = content[snippet_start:snippet_end]
                
                results.append({
                    "document_id": doc_id,
                    "file_path": doc["file_path"],
                    "score": score,
                    "snippet": snippet,
                    "word_count": doc["word_count"]
                })
        
        # 按分数排序并返回前top_k个结果
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

class EnhancedMCPServer:
    """增强版MCP服务器"""
    
    def __init__(self):
        self.doc_processor = SimpleDocumentProcessor()
        self.session_contexts = {}  # 存储会话上下文
        
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
                "name": "validate_environment", 
                "description": "验证服务器环境配置",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "process_document",
                "description": "处理并索引学术文档（支持文本文件）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "要处理的文档文件路径"
                        },
                        "document_name": {
                            "type": "string",
                            "description": "可选的文档名称",
                            "default": ""
                        }
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "query_documents",
                "description": "查询已处理的文档（简单RAG实现）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "要查询的问题或关键词"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "返回的最大结果数量",
                            "default": 3,
                            "minimum": 1,
                            "maximum": 10
                        },
                        "session_id": {
                            "type": "string",
                            "description": "可选的会话ID用于上下文",
                            "default": ""
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
            },
            {
                "name": "get_document_content",
                "description": "获取指定文档的详细内容",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "文档ID"
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": "返回的最大字符数",
                            "default": 1000,
                            "minimum": 100,
                            "maximum": 5000
                        }
                    },
                    "required": ["document_id"]
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
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "documents_processed": len(self.doc_processor.processed_documents)
        }
        
        # 检查API密钥
        api_key = os.environ.get('OPENAI_API_KEY', '')
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
            elif tool_name == "validate_environment":
                return await self.handle_validate_environment(request_id, arguments)
            elif tool_name == "process_document":
                return await self.handle_process_document(request_id, arguments)
            elif tool_name == "query_documents":
                return await self.handle_query_documents(request_id, arguments)
            elif tool_name == "list_documents":
                return await self.handle_list_documents(request_id, arguments)
            elif tool_name == "get_document_content":
                return await self.handle_get_document_content(request_id, arguments)
            else:
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [{
                            "type": "text",
                            "text": f"❌ 未知工具: {tool_name}\\n可用工具: {', '.join([t['name'] for t in self.tools])}"
                        }]
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
    
    async def handle_test_connection(self, request_id: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """处理连接测试"""
        message = arguments.get("message", "Hello MCP!")
        
        result_text = f"""✅ MCP Academic RAG Server (Enhanced) 连接成功!
📩 收到消息: {message}
🕐 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
🐍 Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}
📁 工作目录: {os.getcwd()}
📚 已处理文档: {len(self.doc_processor.processed_documents)}
🔧 可用功能: 文档处理, RAG查询, 内容检索"""
        
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
    
    async def handle_validate_environment(self, request_id: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """处理环境验证"""
        env_status = self.validate_environment()
        
        result_text = f"""🔍 环境验证结果:

✅ API密钥: {'有效' if env_status['api_key_valid'] else '无效'} (长度: {env_status['api_key_length']})
✅ 数据目录: {'存在' if env_status['data_path_exists'] else '不存在'}
🐍 Python版本: {env_status['python_version']}
📁 工作目录: {env_status['working_directory']}
📚 已处理文档: {env_status['documents_processed']}
🕐 检查时间: {env_status['timestamp']}"""
        
        if 'data_path' in env_status:
            result_text += f"\\n📂 数据路径: {env_status['data_path']}"
        if 'data_path_error' in env_status:
            result_text += f"\\n❌ 数据路径错误: {env_status['data_path_error']}"
        
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
        """处理文档处理请求"""
        file_path = arguments.get("file_path", "")
        document_name = arguments.get("document_name", "")
        
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
        
        # 检查文件类型
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in ['.txt', '.md', '.rst', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv']:
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [{
                        "type": "text",
                        "text": f"❌ 暂不支持的文件类型: {file_ext}\\n支持的类型: .txt, .md, .rst, .py, .js, .html, .css, .json, .xml, .csv"
                    }]
                },
                "id": request_id
            }
        
        # 处理文档
        result = self.doc_processor.process_text_file(file_path)
        
        if result["success"]:
            result_text = f"""✅ 文档处理成功!

📄 文件路径: {file_path}
📝 文档名称: {document_name if document_name else os.path.basename(file_path)}
🆔 文档ID: {result['document_id']}
📊 统计信息:
  • 字符数: {result['char_count']:,}
  • 单词数: {result['word_count']:,}
  • 文件类型: {file_ext}

文档已成功处理并建立索引，可以使用 query_documents 工具进行查询。"""
        else:
            result_text = f"""❌ 文档处理失败:

📄 文件路径: {file_path}
💬 错误信息: {result['error']}"""
        
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
        """处理文档查询请求"""
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 3)
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
        
        if len(self.doc_processor.processed_documents) == 0:
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [{
                        "type": "text",
                        "text": "❌ 没有已处理的文档。请先使用 process_document 工具处理一些文档。"
                    }]
                },
                "id": request_id
            }
        
        # 执行查询
        results = self.doc_processor.search_documents(query, max_results)
        
        if not results:
            result_text = f"""🔍 查询结果:

❓ 查询: {query}
📊 结果: 未找到相关文档

请尝试:
• 使用更通用的关键词
• 检查拼写
• 确认相关内容已被处理"""
        else:
            result_text = f"""🔍 查询结果:

❓ 查询: {query}
📊 找到 {len(results)} 个相关文档:

"""
            for i, result in enumerate(results, 1):
                result_text += f"""📄 结果 {i}:
  • 文件: {os.path.basename(result['file_path'])}
  • 相关性: {result['score']} 分
  • 字数: {result['word_count']:,}
  • 预览: {result['snippet'][:100]}...
  • 文档ID: {result['document_id'][:8]}

"""
        
        # 保存会话上下文
        if session_id:
            if session_id not in self.session_contexts:
                self.session_contexts[session_id] = []
            self.session_contexts[session_id].append({
                "query": query,
                "results_count": len(results),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
        
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
    
    async def handle_list_documents(self, request_id: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """处理文档列表请求"""
        docs = self.doc_processor.processed_documents
        
        if not docs:
            result_text = "📚 文档列表:\n\n暂无已处理的文档。使用 process_document 工具来处理文档。"
        else:
            result_text = f"📚 文档列表 (共 {len(docs)} 个):\n\n"
            
            for i, (doc_id, doc) in enumerate(docs.items(), 1):
                result_text += f"""{i}. 📄 {os.path.basename(doc['file_path'])}
   🆔 ID: {doc_id[:8]}...
   📊 {doc['word_count']:,} 词, {doc['char_count']:,} 字符
   🕐 处理时间: {doc['processed_time']}
   📁 路径: {doc['file_path']}

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
    
    async def handle_get_document_content(self, request_id: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """处理获取文档内容请求"""
        document_id = arguments.get("document_id", "")
        max_chars = arguments.get("max_chars", 1000)
        
        if not document_id:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32602,
                    "message": "缺少必需参数: document_id"
                },
                "id": request_id
            }
        
        # 查找文档
        doc = None
        for doc_id, doc_data in self.doc_processor.processed_documents.items():
            if doc_id.startswith(document_id) or doc_id == document_id:
                doc = doc_data
                break
        
        if not doc:
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [{
                        "type": "text",
                        "text": f"❌ 未找到文档 ID: {document_id}"
                    }]
                },
                "id": request_id
            }
        
        # 提取内容
        content = doc["content"]
        if len(content) > max_chars:
            content = content[:max_chars] + f"\\n\\n[内容已截断，显示前{max_chars}字符，总长度: {len(doc['content'])}]"
        
        result_text = f"""📄 文档详情:

🆔 文档ID: {doc['id']}
📁 文件路径: {doc['file_path']}
📊 统计: {doc['word_count']:,} 词, {doc['char_count']:,} 字符
🕐 处理时间: {doc['processed_time']}

📝 内容:
{content}"""
        
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
                    "name": "academic-rag-server-enhanced",
                    "version": "1.0.0-enhanced"
                }
            },
            "id": request_id
        }
    
    async def run(self):
        """运行MCP服务器"""
        logger.info("Starting MCP Academic RAG Server (Enhanced)")
        
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
        server = EnhancedMCPServer()
        env_status = server.validate_environment()
        
        if env_status["api_key_valid"]:
            print("✅ 增强服务器环境验证通过", file=sys.stderr)
            sys.exit(0)
        else:
            print("❌ API密钥验证失败", file=sys.stderr)
            sys.exit(1)
    else:
        # MCP服务器模式
        server = EnhancedMCPServer()
        asyncio.run(server.run())

if __name__ == "__main__":
    main()