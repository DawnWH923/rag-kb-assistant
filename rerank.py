"""
rerank.py — Cross-Encoder 精排
向量检索是"双塔"结构：问题和文档各自独立编码，快但粗糙；
Cross-Encoder 把（问题, 文档）拼接后一起送入模型打相关分，慢但准得多。
流水线：向量召回 top_k(8) → 精排取 top_n(4) → 送 LLM。
首次运行会自动下载 BAAI/bge-reranker-base（约 1.1GB，可用 HF_ENDPOINT 换镜像加速）。
"""
from sentence_transformers import CrossEncoder

_model = None  # 延迟加载，进程内只加载一次


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder("BAAI/bge-reranker-base")
    return _model


def rerank(question: str, texts: list, top_n: int = 4) -> list:
    """按与 question 的相关度对 texts 降序排序，返回前 top_n 条文本"""
    if not texts:
        return []
    pairs = [(question, t) for t in texts]
    scores = _get_model().predict(pairs)
    ranked = sorted(zip(texts, scores), key=lambda x: x[1], reverse=True)
    return [t for t, _ in ranked[:top_n]]
