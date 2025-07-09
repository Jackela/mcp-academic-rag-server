"""
文档分类处理器模块 - 提供文档自动分类功能

该模块实现了一个处理器，用于对学术文献进行自动分类，
生成主题标签，并提取关键内容。
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

from models.document import Document
from models.process_result import ProcessResult
from processors.base_processor import BaseProcessor
from utils.text_utils import TextCleaner, KeywordExtractor
from connectors.api_connector import APIConnector

# 配置日志
logger = logging.getLogger(__name__)

class ClassificationProcessor(BaseProcessor):
    """
    文档分类处理器，用于对学术文献进行分类和标签生成。
    
    该处理器使用API连接器调用文本分类服务，对文档内容进行分析，
    生成主题分类和关键标签，并将结果存储到文档元数据中。
    """
    
    def __init__(self, api_connector: APIConnector, config: Dict[str, Any] = None):
        """
        初始化ClassificationProcessor对象。
        
        Args:
            api_connector: API连接器，用于调用分类服务
            config: 处理器配置，默认为空字典
        """
        super().__init__(
            name="ClassificationProcessor",
            description="对学术文献进行自动分类和标签生成",
            config=config or {}
        )
        self.api_connector = api_connector
        self.categories = self.config.get("categories", [])
        self.max_content_length = self.config.get("max_content_length", 4000)
        self.local_keyword_extraction = self.config.get("local_keyword_extraction", True)
        self.prompt_template = self.config.get("prompt_template", self._get_default_prompt_template())
    
    def process(self, document: Document) -> ProcessResult:
        """
        处理文档并返回处理结果。
        
        从文档中提取内容，调用API进行分类，然后将结果存储到文档中。
        
        Args:
            document: 要处理的Document对象
            
        Returns:
            表示处理结果的ProcessResult对象
        """
        try:
            # 获取文档内容
            content = self._get_document_content(document)
            if not content:
                return ProcessResult.error_result("文档内容为空，无法进行分类")
            
            # 准备分类内容
            prepared_content = self._prepare_content_for_classification(content)
            
            # 调用API进行分类
            classification_result = self._classify_document(prepared_content)
            
            # 处理分类结果
            self._process_classification_result(document, classification_result)
            
            # 本地关键词提取（如果启用）
            if self.local_keyword_extraction:
                self._extract_local_keywords(document, content)
            
            return ProcessResult.success_result(
                "文档分类成功",
                {
                    "categories": document.metadata.get("categories", []),
                    "tags": document.tags,
                    "keywords": document.metadata.get("keywords", [])
                }
            )
            
        except Exception as e:
            logger.error(f"文档分类处理失败: {str(e)}", exc_info=True)
            return ProcessResult.error_result(f"文档分类失败: {str(e)}", e)
    
    def _get_document_content(self, document: Document) -> str:
        """
        获取文档的内容。
        
        优先使用已处理的OCR文本内容，如不存在则尝试获取其他内容。
        
        Args:
            document: Document对象
            
        Returns:
            文档内容文本
        """
        # 尝试从不同处理阶段获取内容
        content_stages = ["ocr", "structure", "preprocessed"]
        
        for stage in content_stages:
            content = document.get_content(stage)
            if content:
                if isinstance(content, str):
                    return content
                elif isinstance(content, dict) and content.get("text"):
                    return content.get("text")
        
        # 如果没有找到内容，记录警告
        logger.warning(f"未找到文档内容: {document.document_id}")
        return ""
    
    def _prepare_content_for_classification(self, content: str) -> str:
        """
        准备用于分类的内容。
        
        清理文本并截取适当长度，以便进行API调用。
        
        Args:
            content: 原始文档内容
            
        Returns:
            处理后的内容
        """
        # 清理文本
        cleaned_content = TextCleaner.clean_for_classification(content)
        
        # 截取适当长度
        if len(cleaned_content) > self.max_content_length:
            # 提取前1/3和后1/3的内容，加上中间的摘要部分
            first_part = cleaned_content[:self.max_content_length // 3]
            
            # 尝试提取摘要部分
            abstract_match = None
            for pattern in [r'Abstract[:\s]*(.*?)(?=\n\s*\n|Introduction)', 
                          r'摘要[：:\s]*(.*?)(?=\n\s*\n|引言|关键词)']:
                matches = re.search(pattern, cleaned_content, re.DOTALL | re.IGNORECASE)
                if matches:
                    abstract_match = matches.group(1).strip()
                    break
            
            middle_part = abstract_match or cleaned_content[len(cleaned_content) // 2 - self.max_content_length // 6:
                                                          len(cleaned_content) // 2 + self.max_content_length // 6]
            
            last_part = cleaned_content[-self.max_content_length // 3:]
            
            return f"{first_part}\n...\n{middle_part}\n...\n{last_part}"
        
        return cleaned_content
    
    def _classify_document(self, content: str) -> Dict[str, Any]:
        """
        调用API对文档进行分类。
        
        构建API请求，调用分类服务，并返回分类结果。
        
        Args:
            content: 处理后的文档内容
            
        Returns:
            分类结果字典
        """
        # 构建提示
        prompt = self._build_classification_prompt(content)
        
        # 调用API
        try:
            response = self.api_connector.make_request(
                method="POST",
                endpoint="chat/completions",
                json_data={
                    "model": self.config.get("model", "gpt-4"),
                    "messages": [
                        {"role": "system", "content": "你是一个专业的学术文献分类助手，负责对文档进行分类、提取关键词和主题。"},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "json_object"}
                }
            )
            
            # 提取分类结果
            if "choices" in response and len(response["choices"]) > 0:
                result_text = response["choices"][0]["message"]["content"]
                try:
                    return json.loads(result_text)
                except json.JSONDecodeError:
                    logger.error(f"API返回的结果不是有效的JSON: {result_text}")
                    return self._extract_json_from_text(result_text)
            else:
                logger.error(f"API响应格式异常: {response}")
                return {}
                
        except Exception as e:
            logger.error(f"调用分类API失败: {str(e)}")
            raise
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        从文本中提取JSON对象。
        
        尝试从可能包含其他内容的文本中提取JSON部分。
        
        Args:
            text: 包含JSON的文本
            
        Returns:
            提取的JSON对象，失败时返回空字典
        """
        try:
            # 尝试查找JSON对象的开始和结束
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx+1]
                return json.loads(json_str)
            
            return {}
        except Exception:
            logger.error("从文本中提取JSON失败")
            return {}
    
    def _process_classification_result(self, document: Document, result: Dict[str, Any]) -> None:
        """
        处理分类结果并更新文档。
        
        将API返回的分类结果存储到文档的元数据和标签中。
        
        Args:
            document: 要更新的Document对象
            result: 分类结果字典
        """
        # 提取分类
        categories = result.get("categories", [])
        if isinstance(categories, str):
            categories = [cat.strip() for cat in categories.split(",")]
        
        # 提取关键词
        keywords = result.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [kw.strip() for kw in keywords.split(",")]
        
        # 提取主题
        topics = result.get("topics", [])
        if isinstance(topics, str):
            topics = [topic.strip() for topic in topics.split(",")]
        
        # 提取摘要
        summary = result.get("summary", "")
        
        # 更新文档元数据
        document.add_metadata("categories", categories)
        document.add_metadata("keywords", keywords)
        document.add_metadata("topics", topics)
        document.add_metadata("summary", summary)
        
        # 更新文档标签（合并类别和主题）
        for tag in categories + topics:
            document.add_tag(tag)
        
        # 存储完整分类结果
        document.store_content("classification", result)
    
    def _extract_local_keywords(self, document: Document, content: str) -> None:
        """
        使用本地算法提取关键词。
        
        作为API分类的补充，使用本地文本处理工具提取额外的关键词。
        
        Args:
            document: Document对象
            content: 文档内容
        """
        try:
            # 提取N元词组
            ngrams = KeywordExtractor.extract_ngrams(content, n=2, min_freq=3)
            ngram_keywords = [ngram for ngram, _ in ngrams[:10]]
            
            # 提取引用
            citations = KeywordExtractor.identify_citations(content)
            
            # 提取参考文献
            references = KeywordExtractor.extract_references(content)
            
            # 更新文档元数据
            current_keywords = document.metadata.get("keywords", [])
            document.add_metadata("keywords", list(set(current_keywords + ngram_keywords)))
            document.add_metadata("citations", citations)
            document.add_metadata("references", references)
            
        except Exception as e:
            logger.warning(f"本地关键词提取失败: {str(e)}")
    
    def _build_classification_prompt(self, content: str) -> str:
        """
        构建分类提示。
        
        根据配置的提示模板和分类类别，生成用于API请求的提示文本。
        
        Args:
            content: 文档内容
            
        Returns:
            分类提示文本
        """
        # 获取分类类别
        categories_str = ", ".join(self.categories) if self.categories else "自动确定合适的分类"
        
        # 替换提示模板中的变量
        prompt = self.prompt_template.replace("{CONTENT}", content)
        prompt = prompt.replace("{CATEGORIES}", categories_str)
        
        return prompt
    
    def _get_default_prompt_template(self) -> str:
        """
        获取默认的提示模板。
        
        Returns:
            默认提示模板
        """
        return """请分析以下学术文献内容，并提供以下信息：
1. 文档类别：请从[{CATEGORIES}]中选择最合适的类别，如果没有合适的，请提供你认为最准确的类别。
2. 关键词：提取5-10个最能代表文档内容的关键词或术语。
3. 主题：确定1-3个文档的主要主题或研究领域。
4. 摘要：用100-200字概括文档的主要内容和贡献。

请以JSON格式返回结果，包含以下字段：categories(数组)、keywords(数组)、topics(数组)、summary(字符串)。

文档内容：
{CONTENT}"""


class TagGenerationProcessor(BaseProcessor):
    """
    标签生成处理器，用于从文档中提取关键词和主题标签。
    
    该处理器使用本地文本处理工具和算法，对文档内容进行分析，
    提取关键词、主题标签、引用和参考文献，并将结果存储到文档中。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化TagGenerationProcessor对象。
        
        Args:
            config: 处理器配置，默认为空字典
        """
        super().__init__(
            name="TagGenerationProcessor",
            description="从学术文献中提取关键词和主题标签",
            config=config or {}
        )
        self.academic_terms_file = self.config.get("academic_terms_file", "")
        self.academic_terms = self._load_academic_terms()
    
    def process(self, document: Document) -> ProcessResult:
        """
        处理文档并返回处理结果。
        
        从文档中提取内容，使用本地算法提取关键词和标签，然后将结果存储到文档中。
        
        Args:
            document: 要处理的Document对象
            
        Returns:
            表示处理结果的ProcessResult对象
        """
        try:
            # 获取文档内容
            content = self._get_document_content(document)
            if not content:
                return ProcessResult.error_result("文档内容为空，无法生成标签")
            
            # 提取N元词组作为关键词
            ngrams = KeywordExtractor.extract_ngrams(content, n=2, min_freq=3)
            ngram_keywords = [ngram for ngram, _ in ngrams[:15]]
            
            # 提取学术术语
            academic_keywords = KeywordExtractor.extract_academic_terms(content, self.academic_terms)
            
            # 提取引用
            citations = KeywordExtractor.identify_citations(content)
            
            # 提取参考文献
            references = KeywordExtractor.extract_references(content)
            
            # 提取数学公式
            formulas = FormatConverter.extract_math_formulas(content)
            
            # 合并关键词
            all_keywords = list(set(ngram_keywords + academic_keywords))
            
            # 更新文档元数据和标签
            document.add_metadata("keywords", all_keywords)
            document.add_metadata("citations", citations)
            document.add_metadata("references", references)
            document.add_metadata("math_formulas", formulas)
            
            # 将关键词添加为标签
            for keyword in all_keywords:
                document.add_tag(keyword)
            
            # 存储标签生成结果
            tag_result = {
                "keywords": all_keywords,
                "academic_terms": academic_keywords,
                "citations": citations,
                "references_count": len(references),
                "math_formulas_count": len(formulas)
            }
            document.store_content("tag_generation", tag_result)
            
            return ProcessResult.success_result(
                "标签生成成功",
                {
                    "keywords_count": len(all_keywords),
                    "citations_count": len(citations),
                    "references_count": len(references),
                    "math_formulas_count": len(formulas)
                }
            )
            
        except Exception as e:
            logger.error(f"标签生成处理失败: {str(e)}", exc_info=True)
            return ProcessResult.error_result(f"标签生成失败: {str(e)}", e)
    
    def _get_document_content(self, document: Document) -> str:
        """
        获取文档的内容。
        
        优先使用已处理的OCR文本内容，如不存在则尝试获取其他内容。
        
        Args:
            document: Document对象
            
        Returns:
            文档内容文本
        """
        # 尝试从不同处理阶段获取内容
        content_stages = ["ocr", "structure", "preprocessed"]
        
        for stage in content_stages:
            content = document.get_content(stage)
            if content:
                if isinstance(content, str):
                    return content
                elif isinstance(content, dict) and content.get("text"):
                    return content.get("text")
        
        # 如果没有找到内容，记录警告
        logger.warning(f"未找到文档内容: {document.document_id}")
        return ""
    
    def _load_academic_terms(self) -> Dict[str, List[str]]:
        """
        加载学术术语字典。
        
        从配置的文件中加载学术术语，如果文件不存在则返回默认术语。
        
        Returns:
            学术术语字典
        """
        if self.academic_terms_file and os.path.exists(self.academic_terms_file):
            try:
                with open(self.academic_terms_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载学术术语文件失败: {str(e)}")
        
        # 返回默认的学术术语
        return {
            "计算机科学": [
                "算法", "数据结构", "机器学习", "深度学习", "人工智能",
                "自然语言处理", "计算机视觉", "数据库", "网络安全", "分布式系统"
            ],
            "数学": [
                "微积分", "线性代数", "概率论", "统计学", "微分方程",
                "拓扑学", "数论", "离散数学", "实分析", "复分析"
            ],
            "物理学": [
                "量子力学", "相对论", "热力学", "电磁学", "粒子物理",
                "核物理", "流体力学", "固体物理", "统计物理", "光学"
            ],
            "生物学": [
                "分子生物学", "遗传学", "生态学", "进化论", "细胞生物学",
                "微生物学", "生物化学", "基因组学", "神经科学", "生物信息学"
            ],
            "医学": [
                "临床医学", "病理学", "药理学", "免疫学", "流行病学",
                "内科学", "外科学", "神经科学", "心血管医学", "肿瘤学"
            ],
            "化学": [
                "有机化学", "无机化学", "物理化学", "分析化学", "生物化学",
                "高分子化学", "催化剂", "化学合成", "光谱学", "电化学"
            ],
            "工程学": [
                "电气工程", "机械工程", "土木工程", "化学工程", "材料科学",
                "控制论", "系统工程", "信号处理", "热力学", "流体力学"
            ],
            "经济学": [
                "微观经济学", "宏观经济学", "计量经济学", "行为经济学", "发展经济学",
                "金融经济学", "国际经济学", "劳动经济学", "公共经济学", "产业经济学"
            ],
            "社会学": [
                "社会结构", "社会变迁", "社会心理学", "社会阶层", "社会发展",
                "群体动力学", "组织社会学", "城市社会学", "家庭社会学", "教育社会学"
            ],
            "心理学": [
                "认知心理学", "发展心理学", "社会心理学", "临床心理学", "人格心理学",
                "异常心理学", "教育心理学", "神经心理学", "积极心理学", "工业心理学"
            ]
        }
