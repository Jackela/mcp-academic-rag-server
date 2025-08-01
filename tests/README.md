# 学术文献OCR电子化与智能检索系统 - 测试框架

本目录包含系统的测试框架和测试用例，用于验证系统功能和性能。

## 测试框架概述

测试框架分为三个主要部分：

1. **单元测试**：针对各个组件的独立功能测试
2. **集成测试**：验证不同组件间的协同工作
3. **性能测试**：评估系统在各种场景下的性能表现

## 测试环境准备

在运行测试之前，请先安装项目依赖：

```bash
pip install -r requirements.txt
```

如果需要生成覆盖率报告，还需额外安装：

```bash
pip install coverage
```

## 测试模块

### 文档处理CLI测试 (`test_document_cli.py`)

测试文档处理命令行接口的各项功能，包括：

- CLI初始化和配置加载
- 文档上传和处理
- 文档信息查询
- 文档列表展示
- 文档导出功能
- 文档删除操作
- 错误处理和边界情况

### 聊天对话CLI测试 (`test_chat_cli.py`)

测试聊天对话命令行接口的各项功能，包括：

- 会话创建和管理
- 消息处理和回复
- 会话历史管理
- 会话导出功能
- 会话列表展示
- 错误处理和边界情况

### Web界面测试 (`test_webapp.py`)

测试Web应用界面的各项功能，包括：

- Web应用初始化和路由配置
- 静态页面渲染
- 文档上传和处理请求
- 文档列表展示
- 聊天对话交互
- AJAX请求处理
- 响应式布局和前端组件
- 错误处理和用户反馈
- 会话管理和状态保持

### 系统集成测试 (`test_integration.py`)

测试整个系统的端到端流程，包括：

- 从文档处理到聊天对话的完整工作流
- 命令行参数处理
- Web界面与后端系统交互
- 系统组件间的数据传递
- 性能测试，包括批量文档处理和查询响应时间

## 如何运行测试

### 运行所有测试

```bash
python -m unittest discover tests
```

### 运行特定测试模块

```bash
python -m unittest tests.test_document_cli
python -m unittest tests.test_chat_cli
python -m unittest tests.test_webapp
python -m unittest tests.test_integration
```

### 运行特定测试用例

```bash
python -m unittest tests.test_document_cli.TestDocumentCLI.test_init
```

### 使用详细输出

```bash
python -m unittest discover tests -v
```

## 性能测试

性能测试包含在`test_integration.py`中的`TestPerformance`类中，用于评估：

1. **批量文档处理性能**：测量处理大量文档的速度和资源使用情况
2. **聊天响应时间**：测量系统对不同类型查询的响应时间
3. **Web界面响应性能**：测量Web应用对并发请求的处理能力

运行性能测试：

```bash
python -m unittest tests.test_integration.TestPerformance
```

## 前端测试

Web界面的前端测试使用以下工具：

1. **Flask测试客户端**：测试后端路由和请求处理
2. **Selenium**：测试前端交互和用户界面功能

运行前端测试：

```bash
python -m unittest tests.test_webapp.TestFrontend
```

## 测试覆盖率

要生成测试覆盖率报告，需要安装`coverage`包：

```bash
pip install coverage
```

然后运行：

```bash
coverage run -m unittest discover tests
coverage report -m
```

生成HTML格式的详细报告：

```bash
coverage html
```

## 添加新测试

添加新测试时，请遵循以下准则：

1. 使用`unittest`框架的标准结构
2. 为每个测试方法提供清晰的文档字符串，说明测试目的
3. 使用`setUp`和`tearDown`方法管理测试环境
4. 使用断言验证测试结果
5. 使用模拟对象(Mock)隔离外部依赖
6. 包含正常情况和错误处理的测试用例

示例：

```python
def test_some_feature(self):
    """测试某个特定功能"""
    # 准备测试数据
    test_data = {...}
    
    # 调用被测试的功能
    result = function_under_test(test_data)
    
    # 验证结果
    self.assertEqual(result, expected_result)
```

## 测试数据

测试中使用的临时数据会在`setUp`方法中创建，并在`tearDown`方法中清理。

对于需要预先准备的测试数据，请将其放在`tests/data`目录中（如有必要可创建此目录）。

## 错误报告

如果测试过程中发现系统错误，请记录以下信息：

1. 测试用例名称
2. 错误类型和消息
3. 测试环境（如操作系统、Python版本等）
4. 重现步骤
5. 期望结果和实际结果

## 持续集成

测试框架设计为可与CI/CD流程集成，支持自动化测试和部署。
