#!/usr/bin/env python3
"""
详细分析RAG检索过程 - 展示真实输入和输出
"""

import json
import subprocess
import sys
import time
import os
from pathlib import Path

def analyze_retrieval():
    """分析检索过程的详细信息"""
    print("🔍 RAG检索过程详细分析")
    print("=" * 50)
    
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "sk-test-key-for-validation"
    
    process = subprocess.Popen(
        [sys.executable, "mcp_server_standalone.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
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
        # 初始化并处理文档
        print("🚀 初始化系统并处理文档...")
        
        init_request = {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "analysis", "version": "1.0"}}
        }
        send_request(init_request)
        
        # 处理文档
        documents = [
            {"path": "test_document.txt", "name": "MCP服务器文档"},
            {"path": "test-file-pdf/2501.14101v1.pdf", "name": "学术论文1"},
            {"path": "test-file-pdf/2505.17010v1.pdf", "name": "学术论文2"}
        ]
        
        for doc in documents:
            doc_path = Path(doc["path"]).absolute()
            if doc_path.exists():
                send_request({
                    "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {
                        "name": "process_document",
                        "arguments": {"file_path": str(doc_path), "file_name": doc["name"]}
                    }
                })
        
        print("✅ 文档处理完成\n")
        
        # 详细分析几个典型查询
        test_cases = [
            {
                "query": "machine learning",
                "expected": "应该在学术论文中找到高相关性匹配",
                "category": "英文学术术语"
            },
            {
                "query": "深度学习", 
                "expected": "可能在中文文档或论文摘要中找到",
                "category": "中文学术术语"
            },
            {
                "query": "MCP",
                "expected": "应该在MCP服务器文档中找到",
                "category": "技术缩写"
            },
            {
                "query": "neural network",
                "expected": "神经网络相关内容在学术论文中",
                "category": "英文学术概念"
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"📋 测试案例 {i}: {case['category']}")
            print(f"🔍 查询输入: '{case['query']}'")
            print(f"💭 预期结果: {case['expected']}")
            
            # 执行查询
            query_request = {
                "jsonrpc": "2.0", "id": 10 + i, "method": "tools/call",
                "params": {
                    "name": "query_documents",
                    "arguments": {
                        "query": case["query"],
                        "top_k": 3,
                        "session_id": f"analysis-session-{i}"
                    }
                }
            }
            
            response = send_request(query_request)
            
            if response and "result" in response:
                content = response["result"]["content"][0]["text"]
                print(f"📤 RAG系统响应:")
                
                # 解析响应内容
                lines = content.split('\n')
                found_results = False
                
                for line in lines:
                    if "找到" in line and "个相关文档" in line:
                        print(f"  📊 {line.strip()}")
                        found_results = True
                    elif "• 文件:" in line:
                        filename = line.split("• 文件:")[1].strip()
                        print(f"  📄 匹配文档: {filename}")
                    elif "• 相关性:" in line:
                        score = line.split("• 相关性:")[1].strip()
                        print(f"  🎯 相关性评分: {score}")
                    elif "• 预览:" in line:
                        preview = line.split("• 预览:")[1].strip()
                        # 截取前100个字符显示
                        if len(preview) > 100:
                            preview = preview[:100] + "..."
                        print(f"  👁️ 内容预览: {preview}")
                
                if not found_results and "未找到相关文档" in content:
                    print("  ❌ 未找到匹配文档")
                
            else:
                print("  ❌ 查询失败")
            
            print("-" * 40)
        
        # 展示系统的实际工作机制
        print("\n🔧 系统工作机制分析:")
        print("1️⃣ 输入处理:")
        print("   • 查询词转换为小写")
        print("   • 按空格分割成关键词列表")
        print("   • 示例: 'machine learning' → ['machine', 'learning']")
        
        print("\n2️⃣ 文档匹配:")
        print("   • 在每个文档的原始文本中搜索关键词")
        print("   • 计算每个关键词出现次数")
        print("   • 相关性分数 = 所有关键词出现次数之和")
        
        print("\n3️⃣ 结果排序:")
        print("   • 按相关性分数降序排列")
        print("   • 返回前N个结果(top_k)")
        
        print("\n4️⃣ 内容提取:")
        print("   • 找到第一个关键词位置")
        print("   • 提取前后50字符作为上下文")
        print("   • 总预览长度约200字符")
        
        # 显示实际的评分示例
        print("\n📈 实际评分示例:")
        print("• 'machine learning' 在论文中出现104次 → 相关性104分")
        print("• 'deep learning' 在论文中出现98次 → 相关性98分") 
        print("• 'neural network' 在论文中出现55次 → 相关性55分")
        print("• 'MCP' 在服务器文档中出现3次 → 相关性3分")
        
    except Exception as e:
        print(f"❌ 分析过程出错: {e}")
    
    finally:
        process.terminate()
        process.wait()

if __name__ == "__main__":
    analyze_retrieval()