"""
文本处理工具模块 - 提供文本分析、标签提取和格式转换功能

该模块提供了与学术文献处理相关的各种文本处理功能，包括文本清洗、
标签(关键词)提取、主题识别、格式转换等。
"""

import re
import string
import unicodedata
from typing import List, Dict, Any, Set, Tuple, Optional
import logging

# 配置日志
logger = logging.getLogger(__name__)

class TextCleaner:
    """文本清洗工具类，提供文本预处理功能"""
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """
        移除文本中的多余空白字符
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 处理后的文本
        """
        # 替换多个空白字符为单个空格
        text = re.sub(r'\s+', ' ', text)
        # 移除行首和行尾的空白字符
        return text.strip()
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Unicode标准化处理
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 标准化后的文本
        """
        return unicodedata.normalize('NFKC', text)
    
    @staticmethod
    def remove_special_chars(text: str, keep_chars: str = '') -> str:
        """
        移除特殊字符
        
        Args:
            text (str): 原始文本
            keep_chars (str): 需要保留的特殊字符
            
        Returns:
            str: 处理后的文本
        """
        # 定义要保留的字符
        pattern = r'[^\w\s' + re.escape(keep_chars) + ']'
        # 替换特殊字符为空格
        return re.sub(pattern, ' ', text)
    
    @staticmethod
    def clean_academic_text(text: str) -> str:
        """
        清洗学术文本，保留必要的学术符号
        
        Args:
            text (str): 原始学术文本
            
        Returns:
            str: 清洗后的文本
        """
        # 学术文本中需要保留的特殊字符
        academic_chars = '.,;:()[]{}"-+*/=%$@#&'
        
        # 标准化处理
        text = TextCleaner.normalize_unicode(text)
        
        # 保留学术符号的情况下移除其他特殊字符
        text = TextCleaner.remove_special_chars(text, academic_chars)
        
        # 移除多余空白字符
        text = TextCleaner.remove_extra_whitespace(text)
        
        return text
    
    @staticmethod
    def extract_paragraphs(text: str) -> List[str]:
        """
        从文本中提取段落
        
        Args:
            text (str): 原始文本
            
        Returns:
            List[str]: 段落列表
        """
        # 按照连续的多个换行符分割文本
        paragraphs = re.split(r'\n\s*\n', text)
        # 清理每个段落
        paragraphs = [TextCleaner.remove_extra_whitespace(p) for p in paragraphs if p.strip()]
        return paragraphs

    @staticmethod
    def clean_for_classification(text: str) -> str:
        """
        为文档分类准备文本
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 处理后的文本
        """
        text = TextCleaner.normalize_unicode(text)
        text = TextCleaner.remove_extra_whitespace(text)
        # 保留更多的标点符号，以保持文本的语义完整性
        text = re.sub(r'[^\w\s.,;:()[\]{}"\'-]', ' ', text)
        return text


class KeywordExtractor:
    """关键词提取工具类，提供从文本中提取关键词和主题的功能"""
    
    @staticmethod
    def extract_ngrams(text: str, n: int = 2, min_freq: int = 2) -> List[Tuple[str, int]]:
        """
        从文本中提取N元词组
        
        Args:
            text (str): 原始文本
            n (int): N元词组的N值
            min_freq (int): 最小出现频率
            
        Returns:
            List[Tuple[str, int]]: (N元词组, 频率)的列表
        """
        # 清理文本
        text = TextCleaner.clean_academic_text(text)
        
        # 分词
        words = text.split()
        
        # 生成N元词组
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        
        # 计算词频
        ngram_freq = {}
        for ngram in ngrams:
            ngram_freq[ngram] = ngram_freq.get(ngram, 0) + 1
        
        # 过滤低频词组
        filtered_ngrams = [(ngram, freq) for ngram, freq in ngram_freq.items() if freq >= min_freq]
        
        # 按频率降序排序
        return sorted(filtered_ngrams, key=lambda x: x[1], reverse=True)
    
    @staticmethod
    def extract_academic_terms(text: str, academic_terms_dict: Dict[str, List[str]]) -> List[str]:
        """
        从文本中提取学术术语
        
        Args:
            text (str): 原始文本
            academic_terms_dict (Dict[str, List[str]]): 学术术语词典
            
        Returns:
            List[str]: 提取的术语列表
        """
        # 清理文本
        text = TextCleaner.clean_academic_text(text).lower()
        
        extracted_terms = []
        
        # 遍历领域术语词典
        for domain, terms in academic_terms_dict.items():
            for term in terms:
                term_lower = term.lower()
                # 使用正则表达式匹配完整词
                pattern = r'\b' + re.escape(term_lower) + r'\b'
                if re.search(pattern, text):
                    extracted_terms.append(term)
        
        return list(set(extracted_terms))
    
    @staticmethod
    def identify_citations(text: str) -> List[str]:
        """
        识别文本中的引用标记
        
        Args:
            text (str): 原始文本
            
        Returns:
            List[str]: 引用标记列表
        """
        # 匹配常见的引用格式
        # 1. [1], [2-4], [5,6,7]
        bracket_citations = re.findall(r'\[\d+(?:[-,]\d+)*\]', text)
        
        # 2. (Smith et al., 2020), (Smith and Jones, 2019)
        author_citations = re.findall(r'\([A-Za-z]+(?:\s+et\s+al\.|\s+and\s+[A-Za-z]+)?(?:,\s+\d{4})\)', text)
        
        # 3. Smith et al. (2020)
        inline_citations = re.findall(r'[A-Za-z]+(?:\s+et\s+al\.|\s+and\s+[A-Za-z]+)?\s+\(\d{4}\)', text)
        
        # 合并所有引用
        all_citations = bracket_citations + author_citations + inline_citations
        
        return all_citations
    
    @staticmethod
    def extract_references(text: str) -> List[str]:
        """
        从文本中提取参考文献
        
        Args:
            text (str): 原始文本
            
        Returns:
            List[str]: 参考文献列表
        """
        references = []
        
        # 查找参考文献部分
        ref_section_patterns = [
            r'References\s*\n([\s\S]+)',
            r'Bibliography\s*\n([\s\S]+)',
            r'参考文献\s*\n([\s\S]+)'
        ]
        
        for pattern in ref_section_patterns:
            match = re.search(pattern, text)
            if match:
                ref_section = match.group(1)
                
                # 尝试按编号分割参考文献
                numbered_refs = re.findall(r'^\s*\[\d+\](.+?)(?=^\s*\[\d+\]|\Z)', ref_section, re.MULTILINE | re.DOTALL)
                if numbered_refs:
                    references.extend([ref.strip() for ref in numbered_refs])
                    continue
                
                # 尝试按行分割参考文献
                line_refs = re.findall(r'^[^\n]+\(\d{4}\)[^\n]+', ref_section, re.MULTILINE)
                if line_refs:
                    references.extend([ref.strip() for ref in line_refs])
                    continue
                
                # 基本按段落分割
                para_refs = re.split(r'\n\s*\n', ref_section)
                references.extend([ref.strip() for ref in para_refs if ref.strip()])
        
        return references


class DocumentStructureExtractor:
    """文档结构提取工具类，提供统一的文档结构识别功能"""
    
    @staticmethod
    def extract_structure(text: str, text_by_page: List[str] = None) -> Dict[str, Any]:
        """
        统一的文档结构提取方法
        
        Args:
            text (str): 完整文本
            text_by_page (List[str], optional): 按页面分组的文本
            
        Returns:
            Dict[str, Any]: 识别的结构元素
        """
        structure = {}
        
        # 识别标题
        structure["title"] = DocumentStructureExtractor._extract_title(text, text_by_page or [])
        
        # 识别摘要
        structure["abstract"] = DocumentStructureExtractor._extract_abstract(text)
        
        # 识别关键词
        structure["keywords"] = DocumentStructureExtractor._extract_keywords(text)
        
        # 识别作者
        structure["authors"] = DocumentStructureExtractor._extract_authors(text)
        
        # 识别章节
        structure["sections"] = DocumentStructureExtractor._extract_sections(text)
        
        # 识别表格
        structure["tables"] = DocumentStructureExtractor._extract_tables(text)
        
        # 识别图表
        structure["figures"] = DocumentStructureExtractor._extract_figures(text)
        
        # 识别参考文献
        structure["references"] = DocumentStructureExtractor._extract_references(text)
        
        return structure
    
    @staticmethod
    def _extract_title(text: str, text_by_page: List[str]) -> str:
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
            
            # 标题通常是全大写或者首字母大写
            title_pattern = r'^(?!(?:Abstract|Introduction|Conclusion|References|Bibliography|Acknowledgements))[A-Z][\w\s:,\-]+$'
            for i in range(min(3, len(non_empty_lines))):
                line = non_empty_lines[i]
                if re.match(title_pattern, line, re.MULTILINE):
                    return line
            
            # 如果没有匹配的行，使用第一个非空行
            return candidate
            
        except Exception as e:
            logger.error(f"提取标题失败: {e}")
            return ""
    
    @staticmethod
    def _extract_abstract(text: str) -> str:
        """
        提取文档摘要
        
        Args:
            text (str): 文档文本
            
        Returns:
            str: 提取的摘要
        """
        try:
            # 使用正则表达式查找摘要部分
            abstract_pattern = r'(?i)abstract[:\.\s]+([\s\S]+?)(?=\n\s*(?:[1I]ntroduction|Keywords:|$))'
            abstract_match = re.search(abstract_pattern, text)
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
    
    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
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
    
    @staticmethod
    def _extract_authors(text: str) -> List[str]:
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
    
    @staticmethod
    def _extract_sections(text: str) -> List[Dict[str, Any]]:
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
            section_pattern = r'^\s*(\d+(?:\.\d+)*)\s+([A-Z][^\n]+)$'
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
    
    @staticmethod
    def _extract_tables(text: str) -> List[Dict[str, Any]]:
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
            table_pattern = r'(?i)(?:table)\s+(\d+)[:\.\s]+([^\n]+)'
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
    
    @staticmethod
    def _extract_figures(text: str) -> List[Dict[str, Any]]:
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
            figure_pattern = r'(?i)(?:figure|fig\.)\s+(\d+)[:\.\s]+([^\n]+)'
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
    
    @staticmethod
    def _extract_references(text: str) -> List[str]:
        """
        提取文档参考文献
        
        Args:
            text (str): 文档文本
            
        Returns:
            List[str]: 参考文献列表
        """
        try:
            # 使用正则表达式查找参考文献部分
            refs_pattern = r'(?i)(?:^|\n)references\s*(?:\n|$)([\s\S]+)'
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


class FormatConverter:
    """格式转换工具类，提供文本格式转换功能"""
    
    @staticmethod
    def text_to_markdown(text: str, doc_structure: Dict[str, Any] = None) -> str:
        """
        将纯文本转换为Markdown格式
        
        Args:
            text (str): 原始文本
            doc_structure (Dict[str, Any], optional): 文档结构信息
            
        Returns:
            str: Markdown格式文本
        """
        if not doc_structure:
            # 如果没有提供文档结构，尝试自动识别
            doc_structure = DocumentStructureExtractor.extract_structure(text)
        
        # 分离标题、摘要和正文
        title = doc_structure.get('title', '')
        abstract = doc_structure.get('abstract', '')
        sections = doc_structure.get('sections', [])
        
        # 构建Markdown
        md_content = []
        
        # 添加标题
        if title:
            md_content.append(f"# {title}\n")
        
        # 添加摘要
        if abstract:
            md_content.append("## 摘要\n")
            md_content.append(f"{abstract}\n")
        
        # 添加正文部分
        for section in sections:
            section_title = section.get('title', '')
            section_content = section.get('content', '')
            
            if section_title:
                # 判断标题级别
                if section.get('level', 2) == 1:
                    md_content.append(f"# {section_title}\n")
                elif section.get('level', 2) == 2:
                    md_content.append(f"## {section_title}\n")
                else:
                    md_content.append(f"### {section_title}\n")
            
            if section_content:
                # 处理列表
                lines = section_content.split('\n')
                processed_lines = []
                
                for line in lines:
                    # 检测和转换列表项
                    if re.match(r'^\s*\d+\.\s', line):
                        # 有序列表
                        processed_lines.append(line)
                    elif re.match(r'^\s*[\-\*•]\s', line):
                        # 无序列表
                        processed_lines.append(line)
                    else:
                        processed_lines.append(line)
                
                section_content = '\n'.join(processed_lines)
                md_content.append(f"{section_content}\n")
        
        # 处理数学公式
        md_text = '\n'.join(md_content)
        md_text = FormatConverter._convert_math_formulas(md_text)
        
        # 转换引用和参考文献
        md_text = FormatConverter._convert_citations(md_text)
        
        return md_text
    
    @staticmethod
    def _detect_structure(text: str) -> Dict[str, Any]:
        """
        检测文本的结构（使用统一的结构提取器）
        
        Args:
            text (str): 原始文本
            
        Returns:
            Dict[str, Any]: 文档结构信息
        """
        return DocumentStructureExtractor.extract_structure(text)
    
    @staticmethod
    def _convert_math_formulas(text: str) -> str:
        """
        转换数学公式为Markdown格式
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 处理后的文本
        """
        # 转换行内公式，如 $E=mc^2$
        text = re.sub(r'(\$[^$\n]+\$)', r'\1', text)
        
        # 转换行间公式，如 $$\int_a^b f(x) dx$$
        def replace_block_formula(match):
            formula = match.group(1)
            return f'\n$$\n{formula}\n$$\n'
        
        text = re.sub(r'\$\$(.*?)\$\$', replace_block_formula, text, flags=re.DOTALL)
        
        return text
    
    @staticmethod
    def _convert_citations(text: str) -> str:
        """
        转换引用为Markdown格式
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 处理后的文本
        """
        # 转换引用
        citations = KeywordExtractor.identify_citations(text)
        
        for citation in citations:
            # 加粗引用
            text = text.replace(citation, f'**{citation}**')
        
        return text
    
    @staticmethod
    def extract_math_formulas(text: str) -> List[str]:
        """
        提取文本中的数学公式
        
        Args:
            text (str): 原始文本
            
        Returns:
            List[str]: 数学公式列表
        """
        # 提取行内公式，如 $E=mc^2$
        inline_formulas = re.findall(r'\$([^$\n]+)\$', text)
        
        # 提取行间公式，如 $$\int_a^b f(x) dx$$
        block_formulas = re.findall(r'\$\$(.*?)\$\$', text, re.DOTALL)
        
        return inline_formulas + block_formulas
