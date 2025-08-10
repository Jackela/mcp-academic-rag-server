"""
知识图谱处理器模块 - 从学术文献中提取实体和关系

该模块提供了知识图谱处理器，用于从学术文档中提取实体、关系和概念，
构建结构化的知识表示，支持语义搜索和知识推理。
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, Counter

from processors.base_processor import BaseProcessor
from models.document import Document
from models.process_result import ProcessResult

# 配置日志
logger = logging.getLogger(__name__)


class KnowledgeGraphProcessor(BaseProcessor):
    """知识图谱处理器，从文档中提取实体和关系"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化知识图谱处理器
        
        Args:
            config (dict, optional): 处理器配置
        """
        name = "知识图谱处理器"
        description = "从学术文档中提取实体、关系和概念，构建知识图谱"
        super().__init__(name, description, config)
        
        # 默认配置
        self.default_config = {
            'extract_entities': True,        # 是否提取实体
            'extract_relations': True,       # 是否提取关系
            'extract_concepts': True,        # 是否提取概念
            'min_entity_freq': 2,           # 实体最小出现频率
            'max_entities': 100,            # 最大实体数量
            'max_relations': 50,            # 最大关系数量
            'language': 'auto',             # 文档语言
            # 实体类型定义
            'entity_types': {
                'PERSON': ['author', 'researcher', 'scientist', 'scholar', 'professor'],
                'ORGANIZATION': ['university', 'institute', 'laboratory', 'company', 'corporation'],
                'LOCATION': ['country', 'city', 'region', 'location', 'place'],
                'TECHNOLOGY': ['algorithm', 'method', 'technique', 'approach', 'framework', 'model', 'system'],
                'CONCEPT': ['concept', 'theory', 'principle', 'idea', 'notion'],
                'METRIC': ['accuracy', 'precision', 'recall', 'f1-score', 'performance', 'efficiency'],
                'DATASET': ['dataset', 'data', 'corpus', 'benchmark', 'collection'],
                'TOOL': ['tool', 'software', 'platform', 'library', 'framework']
            },
            # 关系模式定义
            'relation_patterns': [
                # 基于模式的关系提取
                {'pattern': r'(\w+)\s+(?:is|are|was|were)\s+(?:a|an)?\s*(\w+)', 'relation': 'IS_A'},
                {'pattern': r'(\w+)\s+(?:uses|use|utilize|utilized|employs?)\s+(\w+)', 'relation': 'USES'},
                {'pattern': r'(\w+)\s+(?:improves?|enhances?|increases?)\s+(\w+)', 'relation': 'IMPROVES'},
                {'pattern': r'(\w+)\s+(?:based\s+on|builds?\s+on|extends?)\s+(\w+)', 'relation': 'BASED_ON'},
                {'pattern': r'(\w+)\s+(?:compared\s+to|vs\.?|versus)\s+(\w+)', 'relation': 'COMPARED_TO'},
                {'pattern': r'(\w+)\s+(?:achieves?|obtains?|gets?)\s+(\w+)', 'relation': 'ACHIEVES'},
                {'pattern': r'(\w+)\s+(?:affects?|influences?|impacts?)\s+(\w+)', 'relation': 'AFFECTS'},
                {'pattern': r'(\w+)\s+(?:contains?|includes?|comprises?)\s+(\w+)', 'relation': 'CONTAINS'}
            ]
        }
        
        # 合并配置
        self.config = {**self.default_config, **(config or {})}
        
        # 初始化学术术语词典
        self.academic_terms = self._load_academic_terms()
    
    def get_stage(self) -> str:
        """
        获取处理阶段名称
        
        Returns:
            str: 处理阶段名称
        """
        return "knowledge_graph_extraction"
    
    def process(self, document: Document) -> ProcessResult:
        """
        处理文档知识图谱提取
        
        Args:
            document (Document): 要处理的文档对象
            
        Returns:
            ProcessResult: 处理结果
        """
        try:
            logger.info(f"开始知识图谱提取处理文档: {document.file_name}")
            
            # 更新文档状态
            document.update_status("knowledge_graph_processing")
            
            # 获取结构识别结果
            structure_result = document.get_content("structure_recognition")
            if not structure_result:
                return ProcessResult.error_result("无法获取文档结构，知识图谱提取失败")
            
            # 获取文档文本（优先使用结构化文本）
            text = self._get_document_text(document, structure_result)
            
            if not text:
                return ProcessResult.error_result("无法获取文档文本，知识图谱提取失败")
            
            # 构建知识图谱
            knowledge_graph = self._extract_knowledge_graph(text, structure_result)
            
            # 记录处理结果
            result_data = {
                "knowledge_graph": knowledge_graph,
                "statistics": self._calculate_statistics(knowledge_graph)
            }
            
            # 将处理结果保存到文档对象
            document.store_content(self.get_stage(), result_data)
            document.add_metadata("has_knowledge_graph", True)
            
            # 添加实体和关系到元数据
            entities = knowledge_graph.get("entities", {})
            relations = knowledge_graph.get("relations", [])
            
            document.add_metadata("entity_count", len(entities))
            document.add_metadata("relation_count", len(relations))
            
            # 添加主要实体作为标签
            main_entities = self._get_main_entities(entities, limit=5)
            for entity in main_entities:
                document.add_tag(f"entity:{entity}")
            
            logger.info(f"文档知识图谱提取完成: {document.file_name}")
            return ProcessResult.success_result("知识图谱提取完成", result_data)
            
        except Exception as e:
            logger.error(f"文档知识图谱提取失败: {document.file_name}, 错误: {str(e)}", exc_info=True)
            return ProcessResult.error_result(f"知识图谱提取时发生错误: {str(e)}", e)
    
    def _get_document_text(self, document: Document, structure_result: Dict[str, Any]) -> str:
        """
        获取文档文本（结合OCR和结构信息）
        
        Args:
            document (Document): 文档对象
            structure_result (Dict[str, Any]): 结构识别结果
            
        Returns:
            str: 文档文本
        """
        # 优先使用结构化的文本
        text_parts = []
        
        # 添加标题
        title = structure_result.get("structure", {}).get("title", "")
        if title:
            text_parts.append(title)
        
        # 添加摘要
        abstract = structure_result.get("structure", {}).get("abstract", "")
        if abstract:
            text_parts.append(abstract)
        
        # 添加章节内容
        sections = structure_result.get("structure", {}).get("sections", [])
        for section in sections:
            section_title = section.get("title", "")
            section_content = section.get("content", "")
            if section_title:
                text_parts.append(section_title)
            if section_content:
                text_parts.append(section_content)
        
        # 如果没有结构化文本，使用OCR结果
        if not text_parts:
            ocr_result = document.get_content("ocr")
            if ocr_result and ocr_result.get("text"):
                text_parts.append(ocr_result["text"])
        
        return "\n\n".join(text_parts)
    
    def _extract_knowledge_graph(self, text: str, structure_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        从文本中提取知识图谱
        
        Args:
            text (str): 文档文本
            structure_result (Dict[str, Any]): 结构识别结果
            
        Returns:
            Dict[str, Any]: 知识图谱数据
        """
        knowledge_graph = {
            "entities": {},
            "relations": [],
            "concepts": [],
            "metadata": {}
        }
        
        # 提取实体
        if self.config['extract_entities']:
            entities = self._extract_entities(text)
            knowledge_graph["entities"] = entities
        
        # 提取关系
        if self.config['extract_relations']:
            relations = self._extract_relations(text, knowledge_graph["entities"])
            knowledge_graph["relations"] = relations
        
        # 提取概念
        if self.config['extract_concepts']:
            concepts = self._extract_concepts(text)
            knowledge_graph["concepts"] = concepts
        
        # 添加元数据
        knowledge_graph["metadata"] = {
            "extraction_timestamp": None,  # 可以添加时间戳
            "language": self._detect_language(text),
            "total_tokens": len(text.split()),
            "processing_config": {
                "min_entity_freq": self.config['min_entity_freq'],
                "max_entities": self.config['max_entities'],
                "max_relations": self.config['max_relations']
            }
        }
        
        return knowledge_graph
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """
        从文本中提取实体
        
        Args:
            text (str): 文档文本
            
        Returns:
            Dict[str, Any]: 实体字典 {entity_name: entity_info}
        """
        entities = {}
        
        # 预处理文本
        processed_text = self._preprocess_text(text)
        
        # 基于模式的实体提取
        pattern_entities = self._extract_pattern_based_entities(processed_text)
        entities.update(pattern_entities)
        
        # 基于术语词典的实体提取
        term_entities = self._extract_term_based_entities(processed_text)
        entities.update(term_entities)
        
        # 基于频率的实体提取
        freq_entities = self._extract_frequency_based_entities(processed_text)
        entities.update(freq_entities)
        
        # 过滤和排序实体
        filtered_entities = self._filter_entities(entities)
        
        return filtered_entities
    
    def _extract_pattern_based_entities(self, text: str) -> Dict[str, Any]:
        """
        基于模式的实体提取
        
        Args:
            text (str): 处理后的文本
            
        Returns:
            Dict[str, Any]: 实体字典
        """
        entities = {}
        
        # 人名模式 (首字母大写的连续词)
        person_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        person_matches = re.findall(person_pattern, text)
        
        for person in person_matches:
            if len(person) > 50:  # 过滤过长的匹配
                continue
            entities[person] = {
                'type': 'PERSON',
                'confidence': 0.8,
                'frequency': text.count(person),
                'contexts': self._get_entity_contexts(text, person)
            }
        
        # 技术术语模式
        tech_patterns = [
            r'\b[A-Z]{2,}(?:-[A-Z]{2,})*\b',  # 缩写 (CNN, RNN, etc.)
            r'\b\w+(?:-\w+)*\s+(?:algorithm|method|approach|framework|model)\b',
            r'\b(?:deep|machine|artificial)\s+\w+\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 100:  # 过滤过长的匹配
                    continue
                entities[match] = {
                    'type': 'TECHNOLOGY',
                    'confidence': 0.7,
                    'frequency': text.count(match),
                    'contexts': self._get_entity_contexts(text, match)
                }
        
        return entities
    
    def _extract_term_based_entities(self, text: str) -> Dict[str, Any]:
        """
        基于术语词典的实体提取
        
        Args:
            text (str): 处理后的文本
            
        Returns:
            Dict[str, Any]: 实体字典
        """
        entities = {}
        text_lower = text.lower()
        
        for entity_type, terms in self.config['entity_types'].items():
            for term in terms:
                # 使用正则表达式进行精确匹配
                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                matches = re.finditer(pattern, text_lower)
                
                for match in matches:
                    # 获取原始大小写的词
                    original_term = text[match.start():match.end()]
                    
                    if original_term not in entities:
                        entities[original_term] = {
                            'type': entity_type,
                            'confidence': 0.9,
                            'frequency': 0,
                            'contexts': []
                        }
                    
                    entities[original_term]['frequency'] += 1
                    entities[original_term]['contexts'].extend(
                        self._get_entity_contexts(text, original_term, limit=3)
                    )
        
        return entities
    
    def _extract_frequency_based_entities(self, text: str) -> Dict[str, Any]:
        """
        基于频率的实体提取
        
        Args:
            text (str): 处理后的文本
            
        Returns:
            Dict[str, Any]: 实体字典
        """
        entities = {}
        
        # 提取N元词组
        words = text.split()
        
        # 提取2-gram和3-gram
        for n in [2, 3]:
            ngrams = []
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                # 过滤包含停用词或过短的ngrams
                if self._is_valid_ngram(ngram):
                    ngrams.append(ngram)
            
            # 计算频率
            ngram_counts = Counter(ngrams)
            
            # 选择高频ngrams作为候选实体
            for ngram, count in ngram_counts.most_common(50):
                if count >= self.config['min_entity_freq']:
                    entities[ngram] = {
                        'type': 'CONCEPT',
                        'confidence': min(0.5 + count * 0.1, 0.9),
                        'frequency': count,
                        'contexts': self._get_entity_contexts(text, ngram, limit=2)
                    }
        
        return entities
    
    def _extract_relations(self, text: str, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从文本中提取关系
        
        Args:
            text (str): 文档文本
            entities (Dict[str, Any]): 已提取的实体
            
        Returns:
            List[Dict[str, Any]]: 关系列表
        """
        relations = []
        
        # 基于模式的关系提取
        for pattern_config in self.config['relation_patterns']:
            pattern = pattern_config['pattern']
            relation_type = pattern_config['relation']
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                if len(match.groups()) >= 2:
                    subject = match.group(1).strip()
                    obj = match.group(2).strip()
                    
                    # 验证主语和宾语是否为已识别的实体或有效术语
                    if self._is_valid_relation_entity(subject, entities) and \
                       self._is_valid_relation_entity(obj, entities):
                        
                        relation = {
                            'subject': subject,
                            'predicate': relation_type,
                            'object': obj,
                            'confidence': 0.7,
                            'context': match.group(0),
                            'position': match.start()
                        }
                        relations.append(relation)
        
        # 基于共现的关系提取
        cooccurrence_relations = self._extract_cooccurrence_relations(text, entities)
        relations.extend(cooccurrence_relations)
        
        # 去重和排序
        unique_relations = self._deduplicate_relations(relations)
        sorted_relations = sorted(unique_relations, key=lambda x: x['confidence'], reverse=True)
        
        # 限制关系数量
        return sorted_relations[:self.config['max_relations']]
    
    def _extract_cooccurrence_relations(self, text: str, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        基于实体共现提取关系
        
        Args:
            text (str): 文档文本
            entities (Dict[str, Any]): 已提取的实体
            
        Returns:
            List[Dict[str, Any]]: 关系列表
        """
        relations = []
        entity_names = list(entities.keys())
        
        # 在窗口内查找共现的实体对
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            sentence_entities = []
            for entity in entity_names:
                if entity.lower() in sentence.lower():
                    sentence_entities.append(entity)
            
            # 为同一句子中的实体对创建关系
            for i in range(len(sentence_entities)):
                for j in range(i + 1, len(sentence_entities)):
                    entity1 = sentence_entities[i]
                    entity2 = sentence_entities[j]
                    
                    # 基于实体类型推断关系类型
                    relation_type = self._infer_relation_type(
                        entity1, entities[entity1], 
                        entity2, entities[entity2]
                    )
                    
                    if relation_type:
                        relation = {
                            'subject': entity1,
                            'predicate': relation_type,
                            'object': entity2,
                            'confidence': 0.5,
                            'context': sentence.strip(),
                            'position': text.find(sentence)
                        }
                        relations.append(relation)
        
        return relations
    
    def _extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中提取概念
        
        Args:
            text (str): 文档文本
            
        Returns:
            List[Dict[str, Any]]: 概念列表
        """
        concepts = []
        
        # 基于关键短语的概念提取
        concept_patterns = [
            r'\b(?:concept|notion|idea|principle|theory|framework|paradigm)\s+of\s+(\w+(?:\s+\w+)*)\b',
            r'\b(\w+(?:\s+\w+)*)\s+(?:concept|notion|idea|principle|theory|framework|paradigm)\b',
            r'\b(?:the|a|an)\s+(\w+(?:\s+\w+)*)\s+(?:approach|method|technique|strategy)\b'
        ]
        
        for pattern in concept_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                concept_text = match.group(1).strip()
                if len(concept_text) > 3 and len(concept_text) < 50:
                    concept = {
                        'name': concept_text,
                        'type': 'CONCEPT',
                        'confidence': 0.6,
                        'context': match.group(0),
                        'position': match.start()
                    }
                    concepts.append(concept)
        
        return concepts
    
    def _filter_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        过滤和排序实体
        
        Args:
            entities (Dict[str, Any]): 原始实体字典
            
        Returns:
            Dict[str, Any]: 过滤后的实体字典
        """
        # 按置信度和频率排序
        sorted_entities = sorted(
            entities.items(),
            key=lambda x: (x[1]['confidence'], x[1]['frequency']),
            reverse=True
        )
        
        # 限制实体数量
        max_entities = self.config['max_entities']
        filtered = dict(sorted_entities[:max_entities])
        
        # 过滤低频实体
        min_freq = self.config['min_entity_freq']
        final_entities = {
            name: info for name, info in filtered.items()
            if info['frequency'] >= min_freq
        }
        
        return final_entities
    
    def _deduplicate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        去除重复关系
        
        Args:
            relations (List[Dict[str, Any]]): 关系列表
            
        Returns:
            List[Dict[str, Any]]: 去重后的关系列表
        """
        seen = set()
        unique_relations = []
        
        for relation in relations:
            # 创建关系签名用于去重
            signature = (
                relation['subject'].lower(),
                relation['predicate'],
                relation['object'].lower()
            )
            
            if signature not in seen:
                seen.add(signature)
                unique_relations.append(relation)
        
        return unique_relations
    
    def _get_entity_contexts(self, text: str, entity: str, limit: int = 3) -> List[str]:
        """
        获取实体的上下文
        
        Args:
            text (str): 文档文本
            entity (str): 实体名称
            limit (int): 最大上下文数量
            
        Returns:
            List[str]: 上下文列表
        """
        contexts = []
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            if entity.lower() in sentence.lower() and len(contexts) < limit:
                contexts.append(sentence.strip())
        
        return contexts
    
    def _is_valid_ngram(self, ngram: str) -> bool:
        """
        验证ngram是否为有效的候选实体
        
        Args:
            ngram (str): n元词组
            
        Returns:
            bool: 是否有效
        """
        # 简单的验证规则
        if len(ngram) < 5 or len(ngram) > 50:
            return False
        
        # 过滤纯数字或特殊字符
        if re.match(r'^[\d\s\-_.]+$', ngram):
            return False
        
        # 过滤常见停用词组合
        stop_patterns = [
            r'\b(?:the|a|an|and|or|but|in|on|at|to|for|of|with|by)\b',
            r'\b(?:this|that|these|those|it|they|we|you|i)\b'
        ]
        
        for pattern in stop_patterns:
            if re.search(pattern, ngram.lower()):
                return False
        
        return True
    
    def _is_valid_relation_entity(self, entity: str, entities: Dict[str, Any]) -> bool:
        """
        验证实体是否适合作为关系的主语或宾语
        
        Args:
            entity (str): 实体名称
            entities (Dict[str, Any]): 实体字典
            
        Returns:
            bool: 是否有效
        """
        # 检查是否为已识别的实体
        if entity in entities:
            return True
        
        # 检查是否为有效的术语（长度和格式）
        if len(entity) > 2 and len(entity) < 30:
            if re.match(r'^[a-zA-Z][\w\s-]*[a-zA-Z0-9]$', entity):
                return True
        
        return False
    
    def _infer_relation_type(self, entity1: str, entity1_info: Dict[str, Any],
                           entity2: str, entity2_info: Dict[str, Any]) -> Optional[str]:
        """
        基于实体类型推断关系类型
        
        Args:
            entity1 (str): 实体1名称
            entity1_info (Dict[str, Any]): 实体1信息
            entity2 (str): 实体2名称
            entity2_info (Dict[str, Any]): 实体2信息
            
        Returns:
            Optional[str]: 推断的关系类型
        """
        type1 = entity1_info.get('type', 'UNKNOWN')
        type2 = entity2_info.get('type', 'UNKNOWN')
        
        # 基于实体类型的关系推断规则
        relation_rules = {
            ('PERSON', 'ORGANIZATION'): 'AFFILIATED_WITH',
            ('TECHNOLOGY', 'METRIC'): 'ACHIEVES',
            ('TECHNOLOGY', 'DATASET'): 'USES',
            ('PERSON', 'TECHNOLOGY'): 'DEVELOPED',
            ('ORGANIZATION', 'TECHNOLOGY'): 'DEVELOPED',
            ('CONCEPT', 'TECHNOLOGY'): 'IMPLEMENTED_BY',
            ('TECHNOLOGY', 'TECHNOLOGY'): 'RELATED_TO'
        }
        
        # 尝试正向和反向匹配
        if (type1, type2) in relation_rules:
            return relation_rules[(type1, type2)]
        elif (type2, type1) in relation_rules:
            return relation_rules[(type2, type1)]
        
        # 默认关系
        return 'RELATED_TO'
    
    def _detect_language(self, text: str) -> str:
        """
        检测文档语言
        
        Args:
            text (str): 文档文本
            
        Returns:
            str: 检测到的语言代码
        """
        # 简单的语言检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        
        if chinese_chars > len(text) * 0.1:
            return "zh"
        elif english_words > 0:
            return "en"
        else:
            return "unknown"
    
    def _preprocess_text(self, text: str) -> str:
        """
        预处理文本
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 处理后的文本
        """
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符但保留标点
        text = re.sub(r'[^\w\s.,;:()[\]{}"\'-]', ' ', text)
        
        return text.strip()
    
    def _load_academic_terms(self) -> Dict[str, List[str]]:
        """
        加载学术术语词典
        
        Returns:
            Dict[str, List[str]]: 学术术语词典
        """
        # 这里可以从外部文件加载，现在使用内置词典
        return {
            'machine_learning': [
                'neural network', 'deep learning', 'machine learning', 'artificial intelligence',
                'gradient descent', 'backpropagation', 'convolutional', 'recurrent',
                'transformer', 'attention mechanism', 'feature extraction', 'classification',
                'regression', 'clustering', 'reinforcement learning', 'supervised learning',
                'unsupervised learning', 'semi-supervised', 'transfer learning'
            ],
            'nlp': [
                'natural language processing', 'text mining', 'sentiment analysis',
                'named entity recognition', 'part-of-speech tagging', 'parsing',
                'tokenization', 'lemmatization', 'stemming', 'word embedding',
                'language model', 'seq2seq', 'encoder-decoder'
            ],
            'computer_vision': [
                'computer vision', 'image processing', 'object detection', 'image classification',
                'segmentation', 'feature matching', 'optical character recognition',
                'face recognition', 'edge detection', 'image enhancement'
            ]
        }
    
    def _calculate_statistics(self, knowledge_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算知识图谱统计信息
        
        Args:
            knowledge_graph (Dict[str, Any]): 知识图谱数据
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        entities = knowledge_graph.get("entities", {})
        relations = knowledge_graph.get("relations", [])
        concepts = knowledge_graph.get("concepts", [])
        
        # 按类型统计实体
        entity_types = defaultdict(int)
        for entity_info in entities.values():
            entity_types[entity_info.get('type', 'UNKNOWN')] += 1
        
        # 按类型统计关系
        relation_types = defaultdict(int)
        for relation in relations:
            relation_types[relation.get('predicate', 'UNKNOWN')] += 1
        
        statistics = {
            'total_entities': len(entities),
            'total_relations': len(relations),
            'total_concepts': len(concepts),
            'entity_types': dict(entity_types),
            'relation_types': dict(relation_types),
            'avg_entity_confidence': sum(e.get('confidence', 0) for e in entities.values()) / max(len(entities), 1),
            'avg_relation_confidence': sum(r.get('confidence', 0) for r in relations) / max(len(relations), 1)
        }
        
        return statistics
    
    def _get_main_entities(self, entities: Dict[str, Any], limit: int = 5) -> List[str]:
        """
        获取主要实体
        
        Args:
            entities (Dict[str, Any]): 实体字典
            limit (int): 返回实体数量限制
            
        Returns:
            List[str]: 主要实体列表
        """
        # 按置信度和频率排序
        sorted_entities = sorted(
            entities.items(),
            key=lambda x: (x[1].get('confidence', 0), x[1].get('frequency', 0)),
            reverse=True
        )
        
        return [name for name, _ in sorted_entities[:limit]]