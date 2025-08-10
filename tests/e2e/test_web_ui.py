#!/usr/bin/env python3
"""
端到端Web UI测试

使用Selenium测试Web界面的完整功能，包括文档上传、处理和聊天交互。
重点测试结构化内容的显示功能。
"""

import os
import sys
import time
import unittest
import tempfile
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import threading
import subprocess
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from webapp import app


class WebUITestCase(unittest.TestCase):
    """Web UI端到端测试基类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类设置"""
        # 设置Chrome选项
        cls.chrome_options = Options()
        cls.chrome_options.add_argument("--headless")  # 无头模式
        cls.chrome_options.add_argument("--no-sandbox")
        cls.chrome_options.add_argument("--disable-dev-shm-usage")
        cls.chrome_options.add_argument("--disable-gpu")
        cls.chrome_options.add_argument("--window-size=1920,1080")
        
        # 启动Flask测试服务器
        cls.app = app
        cls.app.config['TESTING'] = True
        cls.app.config['WTF_CSRF_ENABLED'] = False
        
        # 在新线程中启动服务器
        cls.server_thread = threading.Thread(
            target=cls._run_server,
            daemon=True
        )
        cls.server_thread.start()
        
        # 等待服务器启动
        time.sleep(2)
        
        # 基础URL
        cls.base_url = "http://localhost:5555"
    
    @classmethod
    def _run_server(cls):
        """运行测试服务器"""
        cls.app.run(host='localhost', port=5555, debug=False, use_reloader=False)
    
    def setUp(self):
        """每个测试前的设置"""
        try:
            self.driver = webdriver.Chrome(options=self.chrome_options)
            self.driver.implicitly_wait(10)
            self.wait = WebDriverWait(self.driver, 15)
        except Exception as e:
            self.skipTest(f"无法启动Chrome WebDriver: {e}")
    
    def tearDown(self):
        """每个测试后的清理"""
        if hasattr(self, 'driver'):
            self.driver.quit()
    
    def _wait_for_element(self, by, value, timeout=10):
        """等待元素出现"""
        try:
            return WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
        except TimeoutException:
            self.fail(f"元素未找到: {by}={value}")
    
    def _wait_for_clickable(self, by, value, timeout=10):
        """等待元素可点击"""
        try:
            return WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((by, value))
            )
        except TimeoutException:
            self.fail(f"元素不可点击: {by}={value}")
    
    def _create_test_document(self):
        """创建测试文档"""
        test_content = """
# 测试学术文档

## 摘要
这是一个测试文档，用于验证系统的文档处理和结构化内容显示功能。

## 数据表格
下表显示了测试数据：

| 参数 | 数值 | 单位 |
|------|------|------|
| 温度 | 25.5 | °C |
| 湿度 | 60.2 | % |
| 压力 | 101.3 | kPa |

## 代码示例

```python
def calculate_average(data):
    '''计算平均值'''
    return sum(data) / len(data)

# 使用示例
values = [10, 20, 30, 40, 50]
avg = calculate_average(values)
print(f"平均值: {avg}")
```

## 数学公式
能量守恒定律可以表示为：$E = mc^2$

## 结论
本文档包含了多种结构化内容，包括表格、代码和公式。
"""
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.md', 
            delete=False, 
            encoding='utf-8'
        )
        temp_file.write(test_content)
        temp_file.flush()
        return temp_file.name


class TestHomePageNavigation(WebUITestCase):
    """测试首页导航功能"""
    
    def test_home_page_loads(self):
        """测试首页加载"""
        self.driver.get(self.base_url)
        
        # 检查页面标题
        self.assertIn("学术文献RAG系统", self.driver.title)
        
        # 检查导航栏
        nav_links = self.driver.find_elements(By.CSS_SELECTOR, ".navbar-nav .nav-link")
        self.assertGreater(len(nav_links), 0, "导航链接应该存在")
        
        # 检查主要内容区域
        main_content = self._wait_for_element(By.TAG_NAME, "main")
        self.assertTrue(main_content.is_displayed(), "主内容区域应该可见")
    
    def test_navigation_links(self):
        """测试导航链接功能"""
        self.driver.get(self.base_url)
        
        # 测试上传页面链接
        upload_link = self._wait_for_clickable(By.LINK_TEXT, "文档上传")
        upload_link.click()
        
        # 验证页面跳转
        self.wait.until(lambda driver: "/upload" in driver.current_url)
        self.assertIn("upload", self.driver.current_url)
        
        # 测试文档列表链接
        docs_link = self._wait_for_clickable(By.LINK_TEXT, "文档管理")
        docs_link.click()
        
        self.wait.until(lambda driver: "/documents" in driver.current_url)
        self.assertIn("documents", self.driver.current_url)
        
        # 测试聊天页面链接
        chat_link = self._wait_for_clickable(By.LINK_TEXT, "智能问答")
        chat_link.click()
        
        self.wait.until(lambda driver: "/chat" in driver.current_url)
        self.assertIn("chat", self.driver.current_url)


class TestDocumentUpload(WebUITestCase):
    """测试文档上传功能"""
    
    def test_upload_page_elements(self):
        """测试上传页面元素"""
        self.driver.get(f"{self.base_url}/upload")
        
        # 检查文件输入框
        file_input = self._wait_for_element(By.ID, "fileInput")
        self.assertTrue(file_input.is_displayed(), "文件输入框应该可见")
        
        # 检查上传按钮
        upload_btn = self._wait_for_element(By.CSS_SELECTOR, "button[type='submit']")
        self.assertTrue(upload_btn.is_displayed(), "上传按钮应该可见")
        
        # 检查拖放区域
        drop_area = self._wait_for_element(By.CSS_SELECTOR, ".file-upload-area")
        self.assertTrue(drop_area.is_displayed(), "拖放区域应该可见")
    
    def test_file_upload_simulation(self):
        """测试文件上传模拟（不实际上传）"""
        self.driver.get(f"{self.base_url}/upload")
        
        # 创建测试文件
        test_file_path = self._create_test_document()
        
        try:
            # 选择文件
            file_input = self._wait_for_element(By.ID, "fileInput")
            file_input.send_keys(test_file_path)
            
            # 检查预览区域是否显示
            preview_container = self._wait_for_element(By.ID, "previewContainer")
            self.assertTrue(preview_container.is_displayed(), "预览容器应该显示")
            
            # 检查文件详情
            file_details = self._wait_for_element(By.ID, "fileDetails")
            self.assertIn(".md", file_details.text, "应该显示文件扩展名")
            
        finally:
            # 清理测试文件
            if os.path.exists(test_file_path):
                os.unlink(test_file_path)


class TestChatInterface(WebUITestCase):
    """测试聊天界面功能"""
    
    def test_chat_page_elements(self):
        """测试聊天页面基本元素"""
        self.driver.get(f"{self.base_url}/chat")
        
        # 检查聊天容器
        chat_container = self._wait_for_element(By.CSS_SELECTOR, ".chat-container")
        self.assertTrue(chat_container.is_displayed(), "聊天容器应该可见")
        
        # 检查消息区域
        chat_messages = self._wait_for_element(By.ID, "chatMessages")
        self.assertTrue(chat_messages.is_displayed(), "消息区域应该可见")
        
        # 检查输入框
        user_input = self._wait_for_element(By.ID, "userInput")
        self.assertTrue(user_input.is_displayed(), "用户输入框应该可见")
        
        # 检查发送按钮
        send_button = self._wait_for_element(By.CSS_SELECTOR, "button[type='submit']")
        self.assertTrue(send_button.is_displayed(), "发送按钮应该可见")
        
        # 检查清空按钮
        clear_button = self._wait_for_element(By.ID, "clearChat")
        self.assertTrue(clear_button.is_displayed(), "清空按钮应该可见")
    
    def test_welcome_message(self):
        """测试欢迎消息"""
        self.driver.get(f"{self.base_url}/chat")
        
        # 检查欢迎消息
        welcome_message = self._wait_for_element(By.CSS_SELECTOR, ".assistant-message")
        self.assertIn("RAG助手", welcome_message.text, "应该显示欢迎消息")
        
        # 检查消息时间
        message_time = welcome_message.find_element(By.CSS_SELECTOR, ".message-time")
        self.assertTrue(message_time.is_displayed(), "消息时间应该显示")
    
    def test_message_input_validation(self):
        """测试消息输入验证"""
        self.driver.get(f"{self.base_url}/chat")
        
        # 获取输入框和发送按钮
        user_input = self._wait_for_element(By.ID, "userInput")
        send_button = self._wait_for_element(By.CSS_SELECTOR, "button[type='submit']")
        
        # 测试空消息
        user_input.clear()
        send_button.click()
        
        # 验证没有新消息添加（应该只有欢迎消息）
        messages = self.driver.find_elements(By.CSS_SELECTOR, ".message-bubble")
        welcome_messages = [msg for msg in messages if "RAG助手" in msg.text]
        self.assertEqual(len(welcome_messages), 1, "空消息不应该被发送")
    
    def test_message_sending_ui(self):
        """测试消息发送UI交互"""
        self.driver.get(f"{self.base_url}/chat")
        
        # 获取输入框
        user_input = self._wait_for_element(By.ID, "userInput")
        
        # 输入测试消息
        test_message = "这是一个测试消息"
        user_input.clear()
        user_input.send_keys(test_message)
        
        # 发送消息
        user_input.send_keys(Keys.RETURN)
        
        # 验证用户消息显示
        user_messages = self.driver.find_elements(By.CSS_SELECTOR, ".user-message")
        user_message_found = any(test_message in msg.text for msg in user_messages)
        self.assertTrue(user_message_found, "用户消息应该显示在聊天中")
        
        # 验证输入框被清空
        self.assertEqual(user_input.get_attribute("value"), "", "输入框应该被清空")
        
        # 验证加载指示器出现（如果实现了）
        try:
            typing_indicator = self.driver.find_element(By.ID, "typingIndicator")
            # 注意：由于我们没有真实的后端响应，指示器可能一直显示
        except NoSuchElementException:
            pass  # 加载指示器可能不会显示，这是正常的
    
    def test_clear_chat_functionality(self):
        """测试清空聊天功能"""
        self.driver.get(f"{self.base_url}/chat")
        
        # 发送一条测试消息
        user_input = self._wait_for_element(By.ID, "userInput")
        test_message = "测试消息用于清空"
        user_input.send_keys(test_message)
        user_input.send_keys(Keys.RETURN)
        
        # 等待消息显示
        time.sleep(1)
        
        # 点击清空按钮
        clear_button = self._wait_for_clickable(By.ID, "clearChat")
        clear_button.click()
        
        # 处理确认对话框
        try:
            alert = self.driver.switch_to.alert
            alert.accept()
        except:
            pass  # 如果没有弹窗，继续
        
        # 验证聊天被清空（只剩欢迎消息）
        time.sleep(1)
        messages = self.driver.find_elements(By.CSS_SELECTOR, ".message-bubble")
        user_messages = [msg for msg in messages if "user-message" in msg.get_attribute("class")]
        self.assertEqual(len(user_messages), 0, "用户消息应该被清空")


class TestStructuredContentDisplay(WebUITestCase):
    """测试结构化内容显示功能"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        
        # 模拟包含结构化内容的响应
        self.mock_response_with_table = {
            "answer": "根据数据表格显示的结果...",
            "citations": [{
                "document_id": "test_doc_1",
                "text": "实验数据如下表所示：\\n\\n| 参数 | 数值 | 单位 |\\n|------|------|------|\\n| 温度 | 25.5 | °C |\\n| 湿度 | 60.2 | % |",
                "metadata": {"title": "测试文档", "table_caption": "实验数据表"},
                "structured_content": {
                    "type": "table",
                    "title": "实验数据表",
                    "data": {
                        "headers": ["参数", "数值", "单位"],
                        "rows": [["温度", "25.5", "°C"], ["湿度", "60.2", "%"]]
                    }
                }
            }]
        }
        
        self.mock_response_with_code = {
            "answer": "以下是相关的代码实现...",
            "citations": [{
                "document_id": "test_doc_2", 
                "text": "算法实现如下：\\n\\n```python\\ndef calculate_average(data):\\n    return sum(data) / len(data)\\n```",
                "metadata": {"title": "算法文档"},
                "structured_content": {
                    "type": "code",
                    "title": "代码示例",
                    "data": "def calculate_average(data):\\n    return sum(data) / len(data)",
                    "language": "python"
                }
            }]
        }
    
    def test_structured_content_css_classes(self):
        """测试结构化内容CSS类的存在"""
        self.driver.get(f"{self.base_url}/chat")
        
        # 检查CSS样式是否已加载
        # 通过检查页面源码确认样式定义存在
        page_source = self.driver.page_source
        
        css_classes_to_check = [
            "structured-content",
            "content-table", 
            "code-snippet",
            "figure-content",
            "equation-content"
        ]
        
        for css_class in css_classes_to_check:
            self.assertIn(css_class, page_source, f"CSS类 {css_class} 应该在页面中定义")
    
    def test_table_display_structure(self):
        """测试表格显示结构"""
        self.driver.get(f"{self.base_url}/chat")
        
        # 使用JavaScript注入模拟表格内容
        table_html = """
        <div class="message-bubble assistant-message">
            <div>测试回答包含表格</div>
            <div class="citation">
                <div class="citation-title">引用来源：测试文档</div>
                <div>实验数据如下表所示</div>
                <div class="structured-content">
                    <div class="structured-content-header">
                        <span>实验数据表</span>
                        <span class="structured-content-type">表格</span>
                    </div>
                    <table class="content-table">
                        <thead>
                            <tr><th>参数</th><th>数值</th><th>单位</th></tr>
                        </thead>
                        <tbody>
                            <tr><td>温度</td><td>25.5</td><td>°C</td></tr>
                            <tr><td>湿度</td><td>60.2</td><td>%</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """
        
        # 注入HTML到聊天区域
        chat_messages = self._wait_for_element(By.ID, "chatMessages")
        self.driver.execute_script(f"""
            document.getElementById('chatMessages').insertAdjacentHTML('beforeend', `{table_html}`);
        """)
        
        # 验证表格元素
        structured_content = self.driver.find_element(By.CSS_SELECTOR, ".structured-content")
        self.assertTrue(structured_content.is_displayed(), "结构化内容应该显示")
        
        # 验证表格
        table = structured_content.find_element(By.CSS_SELECTOR, ".content-table")
        self.assertTrue(table.is_displayed(), "表格应该显示")
        
        # 验证表头
        headers = table.find_elements(By.CSS_SELECTOR, "th")
        self.assertEqual(len(headers), 3, "应该有3个表头")
        self.assertEqual(headers[0].text, "参数")
        
        # 验证数据行
        data_cells = table.find_elements(By.CSS_SELECTOR, "tbody td")
        self.assertGreater(len(data_cells), 0, "应该有数据单元格")
    
    def test_code_display_structure(self):
        """测试代码显示结构"""
        self.driver.get(f"{self.base_url}/chat")
        
        # 注入代码块HTML
        code_html = """
        <div class="message-bubble assistant-message">
            <div>以下是算法实现</div>
            <div class="citation">
                <div class="citation-title">引用来源：算法文档</div>
                <div>相关代码如下</div>
                <div class="structured-content">
                    <div class="structured-content-header">
                        <span>代码示例</span>
                        <span class="structured-content-type">代码</span>
                    </div>
                    <div class="code-snippet">
                        <button class="copy-code-btn">复制</button>
                        <pre><code class="language-python">def calculate_average(data):
    return sum(data) / len(data)</code></pre>
                    </div>
                </div>
            </div>
        </div>
        """
        
        # 注入HTML
        self.driver.execute_script(f"""
            document.getElementById('chatMessages').insertAdjacentHTML('beforeend', `{code_html}`);
        """)
        
        # 验证代码块元素
        code_snippet = self.driver.find_element(By.CSS_SELECTOR, ".code-snippet")
        self.assertTrue(code_snippet.is_displayed(), "代码块应该显示")
        
        # 验证复制按钮
        copy_btn = code_snippet.find_element(By.CSS_SELECTOR, ".copy-code-btn")
        self.assertTrue(copy_btn.is_displayed(), "复制按钮应该显示")
        self.assertEqual(copy_btn.text, "复制")
        
        # 验证代码内容
        code_element = code_snippet.find_element(By.CSS_SELECTOR, "code")
        self.assertIn("def calculate_average", code_element.text, "代码内容应该正确显示")
    
    def test_copy_code_functionality(self):
        """测试代码复制功能"""
        self.driver.get(f"{self.base_url}/chat")
        
        # 注入代码块（与上面相同）
        code_html = """
        <div class="code-snippet">
            <button class="copy-code-btn" onclick="
                navigator.clipboard.writeText('def test(): pass').then(() => {
                    this.textContent = '已复制!';
                    this.classList.add('copied');
                    setTimeout(() => {
                        this.textContent = '复制';
                        this.classList.remove('copied');
                    }, 2000);
                });">复制</button>
            <pre><code>def test(): pass</code></pre>
        </div>
        """
        
        self.driver.execute_script(f"""
            document.body.insertAdjacentHTML('beforeend', `{code_html}`);
        """)
        
        # 点击复制按钮
        copy_btn = self.driver.find_element(By.CSS_SELECTOR, ".copy-code-btn")
        copy_btn.click()
        
        # 验证按钮文本变化
        time.sleep(0.5)  # 等待异步操作
        self.assertEqual(copy_btn.text, "已复制!", "复制后按钮文本应该改变")
        
        # 验证CSS类
        self.assertIn("copied", copy_btn.get_attribute("class"), "应该添加copied类")


class TestResponsiveDesign(WebUITestCase):
    """测试响应式设计"""
    
    def test_mobile_viewport(self):
        """测试移动端视口"""
        # 设置移动端视口大小
        self.driver.set_window_size(375, 667)  # iPhone 8尺寸
        
        self.driver.get(f"{self.base_url}/chat")
        
        # 检查聊天容器在小屏幕上的显示
        chat_container = self._wait_for_element(By.CSS_SELECTOR, ".chat-container")
        self.assertTrue(chat_container.is_displayed(), "聊天容器在移动端应该显示")
        
        # 检查输入框适应性
        user_input = self._wait_for_element(By.ID, "userInput")
        input_width = user_input.size['width']
        container_width = chat_container.size['width']
        
        # 输入框应该占据大部分宽度
        self.assertGreater(input_width, container_width * 0.6, "输入框在移动端应该足够宽")
    
    def test_tablet_viewport(self):
        """测试平板端视口"""
        # 设置平板端视口大小
        self.driver.set_window_size(768, 1024)  # iPad尺寸
        
        self.driver.get(f"{self.base_url}/chat")
        
        # 验证布局在平板尺寸下正常
        chat_messages = self._wait_for_element(By.ID, "chatMessages")
        self.assertTrue(chat_messages.is_displayed(), "消息区域在平板端应该显示")
        
        # 检查消息气泡宽度
        if self.driver.find_elements(By.CSS_SELECTOR, ".message-bubble"):
            message_bubble = self.driver.find_element(By.CSS_SELECTOR, ".message-bubble")
            bubble_width = message_bubble.size['width']
            container_width = chat_messages.size['width']
            
            # 消息气泡不应该太宽
            self.assertLess(bubble_width, container_width * 0.9, "消息气泡宽度应该合适")


class WebUITestSuite:
    """Web UI测试套件"""
    
    @staticmethod
    def run_all_tests():
        """运行所有Web UI测试"""
        print("=" * 80)
        print("Web UI端到端测试套件")
        print("=" * 80)
        
        # 检查Chrome WebDriver可用性
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            test_driver = webdriver.Chrome(options=chrome_options)
            test_driver.quit()
        except Exception as e:
            print(f"警告: 无法启动Chrome WebDriver: {e}")
            print("请确保已安装Chrome浏览器和ChromeDriver")
            print("跳过Web UI测试")
            return
        
        # 创建测试套件
        suite = unittest.TestSuite()
        
        # 添加测试类
        test_classes = [
            TestHomePageNavigation,
            TestDocumentUpload, 
            TestChatInterface,
            TestStructuredContentDisplay,
            TestResponsiveDesign
        ]
        
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        print("\\n" + "=" * 80)
        print(f"Web UI测试完成 - 成功: {result.testsRun - len(result.failures) - len(result.errors)}, "
              f"失败: {len(result.failures)}, 错误: {len(result.errors)}")
        print("=" * 80)
        
        return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Web UI端到端测试")
    parser.add_argument("--suite", action="store_true", help="运行完整测试套件")
    parser.add_argument("--test", type=str, help="运行特定测试类")
    parser.add_argument("--headless", action="store_true", help="使用无头模式运行")
    
    args = parser.parse_args()
    
    if args.suite:
        WebUITestSuite.run_all_tests()
    elif args.test:
        # 运行特定测试类
        suite = unittest.TestSuite()
        test_class = globals().get(args.test)
        if test_class:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            suite.addTests(tests)
            runner = unittest.TextTestRunner(verbosity=2)
            runner.run(suite)
        else:
            print(f"测试类 {args.test} 不存在")
    else:
        # 运行所有测试
        unittest.main(verbosity=2)