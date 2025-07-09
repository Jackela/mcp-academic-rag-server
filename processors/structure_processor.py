
"""
结构识别处理器模块 - 实现文档结构元素识别功能

该模块提供了结构识别处理器，用于识别学术文档中的标题、摘要、章节、表格等结构元素。
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple

from processors.base_processor import BaseProcessor
from models.document import Document
from models.process_result import ProcessResult

# 配置日志
logger = logging.getLogger(__name__)

class StructureProcessor(BaseProcessor):
    """结构识别处理器，识别文档的结构元素"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化结构识别处理器
        
        Args:
            config (dict, optional): 处理器配置
        """
        name = "结构识别处理器"
        description = "识别文档的标题、摘要、章节等结构元素"
        super().__init__(name, description, config)
        
        # 默认配置
        self.default_config = {
            'recognize_title': True,       # 是否识别标题
            'recognize_abstract': True,    # 是否识别摘要
            'recognize_sections': True,    # 是否识别章节
            'recognize_tables': True,      # 是否识别表格
            'recognize_figures': True,     # 是否识别图表
            'recognize_references': True,  # 是否识别参考文献
            'language': 'auto',            # 文档语言，auto为自动检测
            # 正则表达式模式
            'patterns': {
                'title': r'^(?!(?:Abstract|Introduction|Conclusion|References|Bibliography|Acknowledgements))[A-Z][\w\s:,\-]+$',
                'abstract': r'(?i)abstract[:\.\s]+([\s\S]+?)(?=\n\s*(?:[1I]ntroduction|Keywords:|$))',
                'section_heading': r'^\s*(\d+(?:\.\d+)*)\s+([A-Z][^\n]+)$',
                'figure_caption': r'(?i)(?:figure|fig\.)\s+(\d+)[:\.\s]+([^\n]+)',
                'table_caption': r'(?i)(?:table)\s+(\d+)[:\.\s]+([^\n]+)',
                'references': r'(?i)(?:^|\n)references\s*(?:\n|$)([\s\S]+)',
            }
        }
        
        # 合并配置
        self.config = {**self.default_config, **(config or {})}
    
    def get_stage(self) -> str:
        """
        获取处理阶段名称
        
        Returns:
            str: 处理阶段名称
        """
        return "structure_recognition"
    
    def process(self, document: Document) -> ProcessResult:
        """
        处理文档结构识别
        
        Args:
            document (Document): 要处理的文档对象
            
        Returns:
            ProcessResult: 处理结果
        """
        try:
            logger.info(f"开始结构识别处理文档: {document.file_name}")
            
            # 更新文档状态
            document.update_status("structure_processing")
            
            # 获取OCR结果
            ocr_result = document.get_content("ocr")
            
            if not ocr_result or "text" not in ocr_result:
                return ProcessResult.error_result("无法获取OCR文本，结构识别失败")
            
            # 获取OCR文本
            text = ocr_result["text"]
            text_by_page = ocr_result.get("text_by_page", [])
            
            # 识别文档结构
            structure = self._recognize_structure(text, text_by_page)
            
            # 记录处理结果
            result_data = {
                "structure": structure,
                "language": self._detect_language(text),
                "document_type": self._detect_document_type(text, structure)
            }
            
            # 将处理结果保存到文档对象
            document.store_content(self.get_stage(), result_data)
            document.add_metadata("has_structure", True)
            
            # 添加标签
            if "document_type" in result_data:
                document.add_tag(result_data["document_type"])
            if "language" in result_data:
                document.add_tag(result_data["language"])
            
            # 提取关键元数据
            if structure.get("title"):
                document.add_metadata("title", structure["title"])
            if structure.get("abstract"):
                document.add_metadata("abstract", structure["abstract"])
            if structure.get("keywords"):
                document.add_metadata("keywords", structure["keywords"])
            if structure.get("authors"):
                document.add_metadata("authors", structure["authors"])
            
            logger.info(f"文档结构识别完成: {document.file_name}")
            return ProcessResult.success_result("文档结构识别完成", result_data)
            
        except Exception as e:
            logger.error(f"文档结构识别失败: {document.file_name}, 错误: {str(e)}", exc_info=True)
            return ProcessResult.error_result(f"结构识别时发生错误: {str(e)}", e)
    
    def _recognize_structure(self, text: str, text_by_page: List[str]) -> Dict[str, Any]:
        """
        识别文档结构
        
        Args:
            text (str): 完整文本
            text_by_page (List[str]): 按页面分组的文本
            
        Returns:
            Dict[str, Any]: 识别的结构元素
        """
        structure = {}
        
        # 识别标题
        if self.config['recognize_title']:
            structure["title"] = self._extract_title(text, text_by_page)
        
        # 识别摘要
        if self.config['recognize_abstract']:
            structure["abstract"] = self._extract_abstract(text)
        
        # 识别关键词
        structure["keywords"] = self._extract_keywords(text)
        
        # 识别作者
        structure["authors"] = self._extract_authors(text)
        
        # 识别章节
        if self.config['recognize_sections']:
            structure["sections"] = self._extract_sections(text)
        
        # 识别表格
        if self.config['recognize_tables']:
            structure["tables"] = self._extract_tables(text)
        
        # 识别图表
        if self.config['recognize_figures']:
            structure["figures"] = self._extract_figures(text)
        
        # 识别参考文献
        if self.config['recognize_references']:
            structure["references"] = self._extract_references(text)
        
        return structure
    
    def _extract_title(self, text: str, text_by_page: List[str]) -> str:
        """
        提取文档标题
        
        Args:
            text (str): 文档文本
            text_by_page (List[str]): 按页面分组的文本
            
        Returns:
            str: 提取的标题
        """
        try:
            # 策略1: 使用第一页的开头内容作为标题的候选
            first_page = text_by_page[0] if text_by_page else text
            lines = first_page.split('\n')
            
            # 过滤空行
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            
            if not non_empty_lines:
                return ""
            
            # 寻找第一个非空行，通常这是标题
            candidate = non_empty_lines[0]
            
            # 如果候选行太长，可能不是标题
            if len(candidate) > 150:
                return ""
            
            # 使用正则表达式模式匹配标题
            for i in range(min(3, len(non_empty_lines))):
                line = non_empty_lines[i]
                # 标题通常是全大写或者首字母大写
                if re.match(self.config['patterns']['title'], line, re.MULTILINE):
                    return line
            
            # 如果没有匹配的行，使用第一个非空行
            return candidate
            
        except Exception as e:
            logger.error(f"提取标题失败: {e}")
            return ""
    
    def _extract_abstract(self, text: str) -> str:
        """
        提取文档摘要
        
        Args:
            text (str): 文档文本
            
        Returns:
            str: 提取的摘要
        """
        try:
            # 使用正则表达式查找摘要部分
            abstract_match = re.search(self.config['patterns']['abstract'], text)
            if abstract_match:
                return abstract_match.group(1).strip()
            
            # 另一种模式：查找Abstract后面直到Introduction的内容
            abstract_start = text.lower().find("abstract")
            if abstract_start >= 0:
                intro_start = text.lower().find("introduction", abstract_start)
                if intro_start > abstract_start:
                    abstract_text = text[abstract_start+8:intro_start].strip()
                    return abstract_text
                    
            return ""
            
        except Exception as e:
            logger.error(f"提取摘要失败: {e}")
            return ""
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        提取文档关键词
        
        Args:
            text (str): 文档文本
            
        Returns:
            List[str]: 提取的关键词列表
        """
        try:
            # 查找关键词行
            keywords_pattern = r'(?i)keywords?[:\s]+(.*?)(?=\n\s*\n|\n\s*\d\.|\n\s*[A-Z][a-z]+\s*\n|$)'
            keywords_match = re.search(keywords_pattern, text)
            
            if keywords_match:
                keywords_text = keywords_match.group(1).strip()
                # 分割关键词（通常用逗号或分号分隔）
                keywords = re.split(r'[,;]', keywords_text)
                return [kw.strip() for kw in keywords if kw.strip()]
            
            return []
            
        except Exception as e:
            logger.error(f"提取关键词失败: {e}")
            return []
    
    def _extract_authors(self, text: str) -> List[str]:
        """
        提取文档作者
        
        Args:
            text (str): 文档文本
            
        Returns:
            List[str]: 提取的作者列表
        """
        try:
            # 查找标题下方的作者行
            # 一般在标题和摘要之间
            title_end = 0
            abstract_start = text.lower().find("abstract")
            
            if abstract_start > 0:
                author_text = text[title_end:abstract_start].strip()
                
                # 使用启发式方法查找作者
                # 通常作者行有特定格式，如名字后跟机构标识（上标数字或符号）
                authors = []
                
                # 尝试查找格式为"Name1, Name2, and Name3"的作者行
                author_line_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:,\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*(?:\s+and\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)?)'
                author_matches = re.findall(author_line_pattern, author_text)
                
                if author_matches:
                    author_line = author_matches[0]
                    # 分割作者
                    if "and" in author_line:
                        parts = author_line.split(" and ")
                        if parts[0].count(",") > 0:
                            authors = [a.strip() for a in parts[0].split(",")]
                            authors.append(parts[1].strip())
                        else:
                            authors = [parts[0].strip(), parts[1].strip()]
                    else:
                        authors = [a.strip() for a in author_line.split(",") if a.strip()]
                
                return authors
            
            return []
            
        except Exception as e:
            logger.error(f"提取作者失败: {e}")
            return []
    
    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        提取文档章节结构
        
        Args:
            text (str): 文档文本
            
        Returns:
            List[Dict[str, Any]]: 章节列表
        """
        try:
            sections = []
            
            # 使用正则表达式查找章节标题
            section_pattern = self.config['patterns']['section_heading']
            section_matches = re.finditer(section_pattern, text, re.MULTILINE)
            
            # 根据匹配结果创建章节结构
            last_end = 0
            for i, match in enumerate(section_matches):
                section_num = match.group(1)
                section_title = match.group(2).strip()
                section_start = match.start()
                
                # 创建章节对象
                section = {
                    "number": section_num,
                    "title": section_title,
                    "level": section_num.count('.') + 1,
                    "position": section_start
                }
                
                sections.append(section)
                last_end = match.end()
            
            # 提取章节内容
            if sections:
                for i in range(len(sections) - 1):
                    start_pos = sections[i]["position"] + len(sections[i]["number"]) + len(sections[i]["title"]) + 1
                    end_pos = sections[i+1]["position"]
                    sections[i]["content"] = text[start_pos:end_pos].strip()
                
                # 处理最后一个章节
                last_section = sections[-1]
                start_pos = last_section["position"] + len(last_section["number"]) + len(last_section["title"]) + 1
                # 查找参考文献部分或文档结尾
                refs_pos = text.lower().find("references", start_pos)
                if refs_pos > start_pos:
                    last_section["content"] = text[start_pos:refs_pos].strip()
                else:
                    last_section["content"] = text[start_pos:].strip()
            
            return sections
            
        except Exception as e:
            logger.error(f"提取章节失败: {e}")
            return []
    
    def _extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """
        提取文档表格
        
        Args:
            text (str): 文档文本
            
        Returns:
            List[Dict[str, Any]]: 表格列表
        """
        try:
            tables = []
            
            # 使用正则表达式查找表格标题
            table_pattern = self.config['patterns']['table_caption']
            table_matches = re.finditer(table_pattern, text, re.IGNORECASE)
            
            for match in table_matches:
                table_num = match.group(1)
                table_caption = match.group(2).strip()
                
                # 创建表格对象
                table = {
                    "number": table_num,
                    "caption": table_caption,
                    "position": match.start()
                }
                
                # 尝试提取表格内容（这需要更复杂的算法）
                # 这里简单实现，假设表格在标题下方的下一行开始
                table_start = match.end()
                next_line_start = text.find('\n', table_start)
                if next_line_start > 0:
                    # 查找表格结束位置（通常是空行或下一个图表标题）
                    table_end = text.find('\n\n', next_line_start)
                    if table_end > next_line_start:
                        table_content = text[next_line_start:table_end].strip()
                        table["content"] = table_content
                
                tables.append(table)
            
            return tables
            
        except Exception as e:
            logger.error(f"提取表格失败: {e}")
            return []
    
    def _extract_figures(self, text: str) -> List[Dict[str, Any]]:
        """
        提取文档图表
        
        Args:
            text (str): 文档文本
            
        Returns:
            List[Dict[str, Any]]: 图表列表
        """
        try:
            figures = []
            
            # 使用正则表达式查找图表标题
            figure_pattern = self.config['patterns']['figure_caption']
            figure_matches = re.finditer(figure_pattern, text, re.IGNORECASE)
            
            for match in figure_matches:
                figure_num = match.group(1)
                figure_caption = match.group(2).strip()
                
                # 创建图表对象
                figure = {
                    "number": figure_num,
                    "caption": figure_caption,
                    "position": match.start()
                }
                
                figures.append(figure)
            
            return figures
            
        except Exception as e:
            logger.error(f"提取图表失败: {e}")
            return []
    
    def _extract_references(self, text: str) -> List[str]:
        """
        提取文档参考文献
        
        Args:
            text (str): 文档文本
            
        Returns:
            List[str]: 参考文献列表
        """
        try:
            # 使用正则表达式查找参考文献部分
            refs_pattern = self.config['patterns']['references']
            refs_match = re.search(refs_pattern, text)
            
            if refs_match:
                refs_text = refs_match.group(1).strip()
                
                # 尝试分割成单独的引用条目
                # 这里使用一个简单的启发式方法：引用条目通常以数字或方括号中的数字开头
                refs_items = re.split(r'\n\s*(?:\[\d+\]|\d+\.)\s+', refs_text)
                
                # 第一项通常是空的（因为分割点是条目的开头）
                if refs_items and not refs_items[0].strip():
                    refs_items = refs_items[1:]
                
                # 如果上面的方法找不到条目，尝试按行分割
                if not refs_items:
                    refs_items = [line.strip() for line in refs_text.split('\n') if line.strip()]
                
                return refs_items
            
            return []
            
        except Exception as e:
            logger.error(f"提取参考文献失败: {e}")
            return []
    
    def _detect_language(self, text: str) -> str:
        """
        检测文档语言
        
        Args:
            text (str): 文档文本
            
        Returns:
            str: 检测到的语言代码
        """
        try:
            # 这里使用一个简单的语言检测方法
            # 在实际应用中，可以使用语言检测库，如langdetect
            
            # 英文检测
            english_words = ['the', 'and', 'of', 'in', 'to', 'is', 'that', 'for', 'this', 'with']
            english_count = sum(1 for word in english_words if f" {word} " in f" {text.lower()} ")
            
            # 中文检测
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            
            if chinese_chars > len(text) * 0.3:
                return "zh"
            elif english_count > 5:
                return "en"
            else:
                # 默认返回英文
                return "en"
                
        except Exception as e:
            logger.error(f"语言检测失败: {e}")
            return "en"  # 默认为英文
    
    def _detect_document_type(self, text: str, structure: Dict[str, Any]) -> str:
        """
        检测文档类型
        
        Args:
            text (str): 文档文本
            structure (Dict[str, Any]): 已识别的结构
            
        Returns:
            str: 文档类型
        """
        try:
            # 判断是否为学术论文
            has_abstract = bool(structure.get("abstract"))
            has_references = bool(structure.get("references"))
            
            # 查找关键词
            keywords = {
                "paper": ["study", "research", "experiment", "analysis", "methodology", "result", "conclusion"],
                "report": ["report", "assessment", "evaluation", "summary", "overview", "finding"],
                "thesis": ["thesis", "dissertation", "doctor", "phd", "master"],
                "patent": ["patent", "invention", "claim", "method of", "system for"]
            }
            
            # 计算各类型关键词出现的次数
            scores = {}
            for doc_type, words in keywords.items():
                scores[doc_type] = sum(1 for word in words if f" {word.lower()} " in f" {text.lower()} ")
            
            # 根据关键词和结构特征判断文档类型
            if has_abstract and has_references:
                if "thesis" in scores and scores["thesis"] > 3:
                    return "thesis"
                elif "patent" in scores and scores["patent"] > 3:
                    return "patent"
                elif "paper" in scores and scores["paper"] > 5:
                    return "academic_paper"
                else:
                    return "academic_paper"  # 默认为学术论文
            elif has_references:
                if scores["report"] > scores["paper"]:
                    return "technical_report"
                else:
                    return "article"
            else:
                return "document"  # 未识别类型
                
        except Exception as e:
            logger.error(f"文档类型检测失败: {e}")
            return "document"  # 默认为普通文档
