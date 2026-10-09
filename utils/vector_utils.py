"""
向量处理相关工具函数。

该模块提供了一系列用于处理文本向量化的工具函数，包括文本分块、
向量计算、向量相似度计算和向量可视化等功能。
"""

import logging
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("vector_utils")


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50, respect_sentences: bool = True) -> List[str]:
    """
    将文本分割成固定大小的块，可选择是否保持句子完整性。

    Args:
        text: 要分块的文本
        chunk_size: 每个块的最大字符数，默认为500
        chunk_overlap: 块之间的重叠字符数，默认为50
        respect_sentences: 是否尊重句子边界，默认为True

    Returns:
        文本块列表
    """
    if not text:
        return []

    # 如果文本长度小于块大小，直接返回
    if len(text) <= chunk_size:
        return [text]

    chunks = []

    if respect_sentences:
        # 按句子分割文本
        sentences = re.split(r"(?<=[.!?。！？])\s+", text)

        current_chunk = ""
        current_chunk_sentences: List[str] = []

        for sentence in sentences:
            # 如果当前句子加上当前块的长度超过块大小，则保存当前块并开始新块
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk)

                # 保留部分句子以实现重叠
                overlap_text = ""
                overlap_sentences: List[str] = []
                remaining_overlap = chunk_overlap

                for s in reversed(current_chunk_sentences):
                    if len(s) <= remaining_overlap:
                        overlap_sentences.insert(0, s)
                        remaining_overlap -= len(s)
                    else:
                        break

                overlap_text = "".join(overlap_sentences)
                current_chunk = overlap_text + sentence
                current_chunk_sentences = overlap_sentences + [sentence]
            else:
                current_chunk += sentence
                current_chunk_sentences.append(sentence)

        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)

    else:
        # 简单按字符数分块
        start = 0
        while start < len(text):
            # 确定当前块的结束位置
            end = start + chunk_size

            # 添加当前块
            chunks.append(text[start:end])

            # 移动到下一个块的起始位置，考虑重叠
            start = end - chunk_overlap

    return chunks


def calculate_vector_similarity(vec1: List[float], vec2: List[float], method: str = "dot") -> float:
    """
    计算两个向量的相似度。

    Args:
        vec1: 第一个向量
        vec2: 第二个向量
        method: 相似度计算方法，支持"dot"（点积）、"cosine"（余弦相似度）和"euclidean"（欧氏距离），默认为"dot"

    Returns:
        相似度分数
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"向量维度不匹配: {len(vec1)} vs {len(vec2)}")

    # 转换为numpy数组
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    if method == "dot":
        return float(np.dot(v1, v2))

    elif method == "cosine":
        # 计算余弦相似度
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))

    elif method == "euclidean":
        # 计算欧氏距离，并转换为相似度（距离越小，相似度越高）
        distance = np.linalg.norm(v1 - v2)
        # 使用高斯核将距离转换为相似度
        return float(math.exp(-distance))

    else:
        raise ValueError(f"不支持的相似度计算方法: {method}")


def average_vectors(vectors: List[List[float]]) -> List[float]:
    """
    计算多个向量的平均值。

    Args:
        vectors: 向量列表

    Returns:
        平均向量
    """
    if not vectors:
        return []

    # 转换为numpy数组
    np_vectors = np.array(vectors)

    # 计算平均值
    avg_vector = np.mean(np_vectors, axis=0)

    return [float(value) for value in avg_vector]


def normalize_vector(vector: List[float]) -> List[float]:
    """
    将向量标准化为单位向量。

    Args:
        vector: 输入向量

    Returns:
        标准化后的向量
    """
    # 转换为numpy数组
    np_vector = np.array(vector)

    # 计算向量的范数
    norm = np.linalg.norm(np_vector)

    # 避免除以零
    if norm == 0:
        return vector

    # 标准化向量
    normalized = np_vector / norm

    return [float(value) for value in normalized]


def find_nearest_neighbors(
    query_vector: List[float],
    vectors: List[List[float]],
    top_k: int = 5,
    similarity_method: str = "cosine",
    threshold: Optional[float] = None,
) -> List[Tuple[int, float]]:
    """
    找出与查询向量最相似的k个向量。

    Args:
        query_vector: 查询向量
        vectors: 向量集合
        top_k: 返回的最相似向量数量，默认为5
        similarity_method: 相似度计算方法，默认为"cosine"
        threshold: 相似度阈值，低于该阈值的结果将被过滤，默认为None

    Returns:
        包含(索引, 相似度)的元组列表，按相似度降序排序
    """
    if not vectors or not query_vector:
        return []

    # 计算查询向量与所有向量的相似度
    similarities = []
    for i, vec in enumerate(vectors):
        try:
            sim = calculate_vector_similarity(query_vector, vec, similarity_method)
            similarities.append((i, sim))
        except Exception as e:
            logger.error(f"计算向量 {i} 的相似度时发生错误: {str(e)}")
            similarities.append((i, -float("inf")))

    # 按相似度降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 应用阈值过滤（如果提供）
    if threshold is not None:
        similarities = [(i, sim) for i, sim in similarities if sim >= threshold]

    # 返回前k个结果
    return similarities[:top_k]


def evaluate_embeddings(
    embeddings: List[List[float]], labels: List[Any], evaluation_method: str = "cluster_coherence"
) -> Dict[str, float]:
    """
    评估嵌入向量的质量。

    Args:
        embeddings: 嵌入向量列表
        labels: 每个嵌入向量对应的标签
        evaluation_method: 评估方法，支持"cluster_coherence"（簇内一致性）,
                          "intra_class_similarity"（类内相似度）,
                          和"inter_class_distance"（类间距离），默认为"cluster_coherence"

    Returns:
        包含评估指标的字典
    """
    if len(embeddings) != len(labels):
        raise ValueError(f"嵌入向量数量 ({len(embeddings)}) 与标签数量 ({len(labels)}) 不匹配")

    groups: Dict[Any, List[List[float]]] = {}
    for vector, label in zip(embeddings, labels):
        groups.setdefault(label, []).append(vector)
    results: Dict[str, float] = {}
    if evaluation_method in {"cluster_coherence", "intra_class_similarity"}:
        values = [_mean_pair_similarity(vectors) for vectors in groups.values() if len(vectors) > 1]
        results["intra_class_similarity"] = sum(values) / len(values) if values else 0.0
    if evaluation_method in {"cluster_coherence", "inter_class_distance"}:
        centroids = [average_vectors(vectors) for vectors in groups.values()]
        distances = [
            1 - calculate_vector_similarity(left, right, "cosine")
            for index, left in enumerate(centroids)
            for right in centroids[index + 1 :]
        ]
        results["inter_class_distance"] = sum(distances) / len(distances) if distances else 0.0
    if evaluation_method == "cluster_coherence":
        intra = results["intra_class_similarity"]
        results["separation_ratio"] = results["inter_class_distance"] / intra if intra > 0 else 0.0
    return results


def _mean_pair_similarity(vectors: List[List[float]]) -> float:
    values = [
        calculate_vector_similarity(left, right, "cosine")
        for index, left in enumerate(vectors)
        for right in vectors[index + 1 :]
    ]
    return sum(values) / len(values) if values else 0.0


def reduce_dimensionality(vectors: List[List[float]], dim: int = 2, method: str = "pca") -> List[List[float]]:
    """
    降低向量维度，用于可视化。

    Args:
        vectors: 高维向量列表
        dim: 目标维度，默认为2
        method: 降维方法，支持"pca"和"tsne"，默认为"pca"

    Returns:
        降维后的向量列表
    """
    if not vectors:
        return []

    try:
        # 转换为numpy数组
        np_vectors = np.array(vectors)

        if method == "pca":
            from sklearn.decomposition import PCA

            reduced_vectors = PCA(n_components=dim).fit_transform(np_vectors)
        elif method == "tsne":
            from sklearn.manifold import TSNE

            reduced_vectors = TSNE(n_components=dim, random_state=42).fit_transform(np_vectors)
        else:
            raise ValueError(f"不支持的降维方法: {method}")

        # 执行降维
        return [[float(value) for value in row] for row in reduced_vectors]

    except Exception as e:
        logger.error(f"降维失败: {str(e)}")
        return []


def optimize_chunk_size(
    text: str,
    min_size: int = 100,
    max_size: int = 1000,
    step: int = 100,
    embedding_fn: Optional[Callable[[str], List[float]]] = None,
) -> Tuple[int, float]:
    """
    通过评估不同块大小的嵌入质量，确定最佳的文本分块大小。

    Args:
        text: 要分析的文本
        min_size: 最小块大小，默认为100
        max_size: 最大块大小，默认为1000
        step: 块大小增加步长，默认为100
        embedding_fn: 生成文本嵌入的函数，默认为None（如果为None则仅评估分块数量）

    Returns:
        包含最佳块大小和评分的元组
    """
    if not text:
        return (min_size, 0.0)

    chunk_sizes = range(min_size, max_size + 1, step)
    best_size = min_size
    best_score = 0.0

    for size in chunk_sizes:
        # 使用当前块大小分块
        chunks = chunk_text(text, chunk_size=size, chunk_overlap=int(size * 0.1))

        # 如果提供了嵌入函数，则评估嵌入质量
        if embedding_fn is not None:
            try:
                score = _chunk_embedding_score(chunks, embedding_fn)
                if score > best_score:
                    best_score, best_size = score, size
            except Exception as error:
                logger.error(f"评估块大小 {size} 时发生错误: {error}")
                continue
        else:
            # 如果没有提供嵌入函数，则仅基于块数量评估
            chunk_count = len(chunks)

            # 理想的块数量范围（可根据需要调整）
            ideal_min = 5
            ideal_max = 15

            if ideal_min <= chunk_count <= ideal_max:
                # 在理想范围内，优先选择较大的块大小（减少块数量）
                if size > best_size:
                    best_size = size
                    best_score = 1.0 - abs((chunk_count - (ideal_min + ideal_max) / 2) / ((ideal_max - ideal_min) / 2))

    return (best_size, best_score)


def _chunk_embedding_score(chunks: List[str], embedding_fn: Callable[[str], List[float]]) -> float:
    embeddings = [embedding_fn(chunk) for chunk in chunks]
    values = [calculate_vector_similarity(left, right, "cosine") for left, right in zip(embeddings, embeddings[1:])]
    average = sum(values) / len(values) if values else 0.0
    count = len(chunks)
    count_factor = count / 3 if count < 3 else 20 / count if count > 20 else 1.0
    return average * count_factor


def get_academic_citation_embeddings(
    text: str,
    citation_pattern: str = r"\[\d+\]|\(\w+,\s+\d{4}\)",
    embedding_fn: Optional[Callable[[str], List[float]]] = None,
) -> Tuple[List[str], List[List[float]]]:
    """
    提取学术文献中的引用及其上下文，并为每个引用上下文生成嵌入向量。

    Args:
        text: 学术文献文本
        citation_pattern: 引用模式的正则表达式，默认匹配[1]或(Author, 2020)格式
        embedding_fn: 生成文本嵌入的函数，默认为None

    Returns:
        包含引用上下文和对应嵌入向量的元组
    """
    if not text or not embedding_fn:
        return ([], [])

    # 查找所有引用
    citations = re.finditer(citation_pattern, text)
    citation_contexts = []
    embeddings = []

    for match in citations:
        # 获取引用位置
        start, end = match.span()

        # 提取引用上下文（引用前后150个字符）
        context_start = max(0, start - 150)
        context_end = min(len(text), end + 150)

        # 提取上下文文本
        context = text[context_start:context_end]

        # 添加到列表
        citation_contexts.append(context)

        # 生成嵌入向量
        try:
            embedding = embedding_fn(context)
            embeddings.append(embedding)
        except Exception as e:
            logger.error(f"为引用上下文生成嵌入向量时发生错误: {str(e)}")
            # 添加空向量以保持与上下文列表的对应关系
            embeddings.append([0.0] * 768)  # 使用默认维度

    return (citation_contexts, embeddings)


def analyze_embedding_distribution(embeddings: List[List[float]]) -> Dict[str, Any]:
    """
    分析嵌入向量的分布特性。

    Args:
        embeddings: 嵌入向量列表

    Returns:
        包含分布特性的字典
    """
    if not embeddings:
        return {}

    # 转换为numpy数组
    np_embeddings = np.array(embeddings, dtype=float)

    # 计算基本统计量
    mean_vector = np.mean(np_embeddings, axis=0)
    std_vector = np.std(np_embeddings, axis=0)
    min_vector = np.min(np_embeddings, axis=0)
    max_vector = np.max(np_embeddings, axis=0)

    # 计算平均范数
    norms = np.linalg.norm(np_embeddings, axis=1)
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)

    # 计算向量间平均相似度
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = calculate_vector_similarity(embeddings[i], embeddings[j], "cosine")
            similarities.append(sim)

    mean_similarity = np.mean(similarities) if similarities else 0.0
    std_similarity = np.std(similarities) if similarities else 0.0

    # 返回分析结果
    return {
        "count": len(embeddings),
        "dimension": len(embeddings[0]),
        "mean_norm": float(mean_norm),
        "std_norm": float(std_norm),
        "mean_similarity": float(mean_similarity),
        "std_similarity": float(std_similarity),
        "min_similarity": float(np.min(similarities)) if similarities else 0.0,
        "max_similarity": float(np.max(similarities)) if similarities else 0.0,
        "value_range": {
            "min": min_vector.tolist(),
            "max": max_vector.tolist(),
            "mean": mean_vector.tolist(),
            "std": std_vector.tolist(),
        },
    }


def filter_embeddings(
    embeddings: List[List[float]], texts: List[str], filter_method: str = "outlier", threshold: float = 2.0
) -> Tuple[List[List[float]], List[str], List[int]]:
    """
    过滤嵌入向量，去除噪声或异常值。

    Args:
        embeddings: 嵌入向量列表
        texts: 对应的文本列表
        filter_method: 过滤方法，支持"outlier"（异常值检测）和"noise"（噪声检测），默认为"outlier"
        threshold: 过滤阈值，默认为2.0

    Returns:
        包含过滤后的嵌入向量、文本和被保留的索引的元组
    """
    if not embeddings or not texts or len(embeddings) != len(texts):
        return ([], [], [])

    if filter_method == "outlier":
        keep_indices = _outlier_indices(embeddings, threshold)
    elif filter_method == "noise":
        keep_indices = _noise_indices(embeddings, threshold)
    else:
        keep_indices = list(range(len(embeddings)))

    # 筛选向量和文本
    filtered_embeddings = [embeddings[i] for i in keep_indices]
    filtered_texts = [texts[i] for i in keep_indices]

    return (filtered_embeddings, filtered_texts, keep_indices)


def _outlier_indices(embeddings: List[List[float]], threshold: float) -> List[int]:
    vectors = np.array(embeddings, dtype=float)
    mean = np.mean(vectors, axis=0)
    distances = [float(np.linalg.norm(vector - mean)) for vector in vectors]
    average, deviation = float(np.mean(distances)), float(np.std(distances))
    return [index for index, distance in enumerate(distances) if abs(distance - average) <= threshold * deviation]


def _noise_indices(embeddings: List[List[float]], threshold: float) -> List[int]:
    keep = []
    for index, vector in enumerate(embeddings):
        values = [
            calculate_vector_similarity(vector, other, "cosine")
            for other_index, other in enumerate(embeddings)
            if index != other_index
        ]
        average = sum(values) / len(values) if values else 0.0
        if average >= threshold:
            keep.append(index)
    return keep


def combine_embeddings(embeddings: List[List[float]], method: str = "average") -> List[float]:
    """
    将多个嵌入向量合并为一个。

    Args:
        embeddings: 嵌入向量列表
        method: 合并方法，支持"average"（平均）、"max"（逐元素最大值）和"weighted"（加权平均），默认为"average"

    Returns:
        合并后的嵌入向量
    """
    if not embeddings:
        return []

    # 转换为numpy数组
    np_embeddings = np.array(embeddings, dtype=float)

    if method == "average":
        # 计算平均值
        combined = np.mean(np_embeddings, axis=0)

    elif method == "max":
        # 计算逐元素最大值
        combined = np.max(np_embeddings, axis=0)

    elif method == "weighted":
        # 计算向量范数
        norms = np.linalg.norm(np_embeddings, axis=1)

        # 计算权重（使用范数的倒数，使较小的向量有更大的权重）
        weights = 1.0 / (norms + 1e-10)  # 添加小值避免除以零
        weights = weights / np.sum(weights)  # 标准化权重

        # 计算加权平均
        combined = np.zeros_like(np_embeddings[0])
        for i, vec in enumerate(np_embeddings):
            combined += weights[i] * vec

    else:
        # 未知的合并方法，使用平均值
        combined = np.mean(np_embeddings, axis=0)

    return [float(value) for value in combined]


def extract_academic_entities(text: str, entity_patterns: Optional[Dict[str, str]] = None) -> Dict[str, List[str]]:
    """
    从学术文献中提取实体。

    Args:
        text: 学术文献文本
        entity_patterns: 实体类型到正则表达式的映射，默认为None

    Returns:
        实体类型到提取实体列表的映射
    """
    if not text:
        return {}

    # 默认实体模式
    default_patterns = {
        "citation": r"\[\d+\]|\(\w+,\s+\d{4}\)",
        "figure": r"图\s*\d+|Figure\s*\d+|Fig\.\s*\d+",
        "table": r"表\s*\d+|Table\s*\d+",
        "equation": r"公式\s*\d+|Equation\s*\d+|Eq\.\s*\d+",
        "section": r"第\s*\d+\s*节|Section\s*\d+",
        "doi": r"DOI:\s*10\.\d{4,}\/[^\s]+",
        "url": r"https?://[^\s]+",
    }

    # 使用提供的模式或默认模式
    patterns = entity_patterns or default_patterns

    # 提取实体
    entities: Dict[str, List[str]] = {}

    for entity_type, pattern in patterns.items():
        entities[entity_type] = []
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            entity_text = match.group(0)
            entities[entity_type].append(entity_text)

    return entities
