#!/usr/bin/env python3
"""
简单的MCP客户端测试脚本
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path


async def test_mcp_server():
    """测试MCP服务器"""

    # 启动服务器进程
    server_path = Path("E:/Code/mcp-academic-rag-server/mcp_server_sdk.py")

    print("🚀 启动MCP服务器...")
    process = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**dict(os.environ), "OPENAI_API_KEY": "sk-test-key-for-validation"},
    )

    def send_request(request):
        """发送请求到服务器"""
        request_json = json.dumps(request)
        print(f"📤 发送: {request_json}")

        process.stdin.write(request_json + "\n")
        process.stdin.flush()

        # 读取响应
        response_line = process.stdout.readline()
        if response_line:
            response = json.loads(response_line.strip())
            print(f"📥 响应: {json.dumps(response, indent=2, ensure_ascii=False)}")
            return response
        return None

    try:
        # 1. 初始化请求
        print("\n1️⃣ 发送初始化请求...")
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }
        init_response = send_request(init_request)

        if init_response and "result" in init_response:
            print("✅ 初始化成功!")

            # 2. 发送通知表示初始化完成
            print("\n2️⃣ 发送初始化完成通知...")
            notification = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
            send_request(notification)

            # 3. 获取工具列表
            print("\n3️⃣ 获取工具列表...")
            tools_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            tools_response = send_request(tools_request)

            if tools_response and "result" in tools_response:
                tools = tools_response["result"].get("tools", [])
                print(f"✅ 找到 {len(tools)} 个工具:")
                for tool in tools:
                    print(f"   • {tool['name']}: {tool['description']}")

                # 4. 测试工具调用
                print("\n4️⃣ 测试 test_connection 工具...")
                call_request = {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "test_connection", "arguments": {"message": "Hello from test client!"}},
                }
                call_response = send_request(call_request)

                if call_response and "result" in call_response:
                    content = call_response["result"].get("content", [])
                    for item in content:
                        if item.get("type") == "text":
                            print(f"✅ 工具响应:\n{item.get('text')}")
                else:
                    print("❌ 工具调用失败")
            else:
                print("❌ 获取工具列表失败")
        else:
            print("❌ 初始化失败")

    except Exception as e:
        print(f"❌ 测试错误: {e}")

    finally:
        # 清理进程
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    import os

    asyncio.run(test_mcp_server())
