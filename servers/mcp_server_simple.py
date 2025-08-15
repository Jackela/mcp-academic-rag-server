#!/usr/bin/env python3
"""
最简化的MCP Academic RAG Server - 绕过复杂依赖
"""

import asyncio
import json
import sys
import os

def validate_api_key(api_key: str) -> bool:
    """验证API密钥格式"""
    return (api_key and 
            isinstance(api_key, str) and 
            api_key.startswith('sk-') and 
            len(api_key) >= 21)

def main():
    """简单的入口点"""
    if len(sys.argv) > 1 and sys.argv[1] == "--validate-only":
        api_key = os.environ.get('OPENAI_API_KEY')
        if validate_api_key(api_key):
            print("✅ 环境验证通过", file=sys.stderr)
            sys.exit(0)
        else:
            print("❌ API密钥无效", file=sys.stderr)
            sys.exit(1)
    
    # MCP协议最简实现
    print('{"jsonrpc": "2.0", "result": {"capabilities": {"tools": {"listChanged": false}}, "serverInfo": {"name": "academic-rag-server", "version": "1.0.0"}}, "id": 0}')
    
    # 保持运行
    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            
            try:
                request = json.loads(line)
                method = request.get("method")
                request_id = request.get("id")
                
                if method == "tools/list":
                    response = {
                        "jsonrpc": "2.0",
                        "result": {
                            "tools": [
                                {
                                    "name": "test_connection",
                                    "description": "测试MCP连接",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {},
                                        "required": []
                                    }
                                }
                            ]
                        },
                        "id": request_id
                    }
                elif method == "tools/call":
                    tool_name = request.get("params", {}).get("name")
                    response = {
                        "jsonrpc": "2.0",
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"✅ 简化MCP服务器连接成功！工具: {tool_name}"
                                }
                            ]
                        },
                        "id": request_id
                    }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "error": {"code": -32601, "message": f"Method not found: {method}"},
                        "id": request_id
                    }
                
                print(json.dumps(response))
                sys.stdout.flush()
                
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
                    "id": request.get("id") if 'request' in locals() else None
                }
                print(json.dumps(error_response))
                sys.stdout.flush()
                
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()