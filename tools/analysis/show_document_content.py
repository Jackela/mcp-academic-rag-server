#!/usr/bin/env python3
"""
展示文档的实际内容片段
"""

import json
import os
import subprocess
import sys
import time


def show_document_content():
    """展示文档内容和查询匹配"""
    print("📄 文档内容和查询匹配展示")
    print("=" * 50)

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "sk-test-key-for-validation"

    process = subprocess.Popen(
        [sys.executable, "mcp_server_standalone.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    time.sleep(1)

    def send_request(request):
        request_json = json.dumps(request)
        process.stdin.write(request_json + "\n")
        process.stdin.flush()

        response_line = process.stdout.readline()
        if response_line.strip():
            return json.loads(response_line.strip())
        return None

    try:
        # 初始化
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "content", "version": "1.0"},
            },
        }
        send_request(init_request)

        # 处理一个PDF文档
        from pathlib import Path

        pdf_path = Path("test-file-pdf/2505.17010v1.pdf").absolute()
        if pdf_path.exists():
            print("📄 处理PDF文档以获取内容...")
            response = send_request(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "process_document",
                        "arguments": {"file_path": str(pdf_path), "file_name": "学术论文样本"},
                    },
                }
            )

            if response and "result" in response:
                print("✅ 文档处理成功")

        # 展示具体的查询和匹配过程
        print("\n🔍 具体查询示例:")

        # 查询"machine learning"
        print("\n1️⃣ 查询: 'machine learning'")
        response = send_request(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "query_documents", "arguments": {"query": "machine learning", "top_k": 1}},
            }
        )

        if response and "result" in response:
            content = response["result"]["content"][0]["text"]
            print("📤 完整响应:")
            print(content)

        print("\n" + "=" * 60)

        # 查询"algorithm"
        print("\n2️⃣ 查询: 'algorithm'")
        response = send_request(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "query_documents", "arguments": {"query": "algorithm", "top_k": 1}},
            }
        )

        if response and "result" in response:
            content = response["result"]["content"][0]["text"]
            print("📤 完整响应:")
            print(content)

        print("\n" + "=" * 60)

        # 查询不存在的词
        print("\n3️⃣ 查询: 'blockchain' (不存在的词)")
        response = send_request(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "query_documents", "arguments": {"query": "blockchain", "top_k": 1}},
            }
        )

        if response and "result" in response:
            content = response["result"]["content"][0]["text"]
            print("📤 完整响应:")
            print(content)

    except Exception as e:
        print(f"❌ 错误: {e}")

    finally:
        process.terminate()
        process.wait()


if __name__ == "__main__":
    show_document_content()
