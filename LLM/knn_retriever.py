# -*- coding: utf-8 -*-
"""
KNN 检索模块 - 从人工标注样本中检索相似示例作为 few-shot
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

from config import ORIGINAL_TRAIN_FILE


class KNNRetriever:
    """基于语义相似度的 KNN 检索器"""
    
    def __init__(self, model_name: str = "m3rg-iitd/matscibert", k: int = 3):
        """
        初始化检索器
        
        Args:
            model_name: sentence-transformer 模型名称
                - "m3rg-iitd/matscibert": 材料科学专用模型（推荐）
                - "allenai/specter2_base": 科学论文专用模型
                - "BAAI/bge-base-en-v1.5": 通用高质量模型
            k: 检索的示例数量
        """
        self.k = k
        print(f"加载嵌入模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # 加载人工标注样本
        self.samples = self._load_labeled_samples()
        
        # 预计算所有样本的嵌入
        self.embeddings = self._compute_embeddings()
        print(f"已加载 {len(self.samples)} 个人工标注样本")
    
    def _load_labeled_samples(self) -> List[Dict]:
        """加载人工标注的训练样本"""
        with open(ORIGINAL_TRAIN_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            # 提取摘要文本
            text = item.get('data', {}).get('text', '')
            if not text:
                continue
            
            # 提取标注的实体
            entities = []
            annotations = item.get('annotations', [])
            if annotations:
                results = annotations[0].get('result', [])
                for r in results:
                    if r.get('type') == 'labels':
                        value = r.get('value', {})
                        entities.append({
                            'text': value.get('text', ''),
                            'label': value.get('labels', [''])[0]
                        })
            
            samples.append({
                'text': text,
                'entities': entities
            })
        
        return samples
    
    def _compute_embeddings(self) -> np.ndarray:
        """预计算所有样本的嵌入向量"""
        texts = [s['text'] for s in self.samples]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings
    
    def retrieve(self, query_text: str, k: int = None) -> List[Dict]:
        """
        检索与查询文本最相似的 k 个样本
        
        Args:
            query_text: 待标注的摘要文本
            k: 返回的样本数量（默认使用初始化时的 k）
            
        Returns:
            相似样本列表，每个包含 text, entities, similarity
        """
        if k is None:
            k = self.k
        
        # 计算查询文本的嵌入
        query_embedding = self.model.encode([query_text], show_progress_bar=False)[0]
        
        # 计算余弦相似度
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # 获取 top-k 索引
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # 构建结果
        results = []
        for idx in top_k_indices:
            results.append({
                'text': self.samples[idx]['text'],
                'entities': self.samples[idx]['entities'],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def format_as_fewshot(self, examples: List[Dict]) -> str:
        """
        将检索到的样本格式化为 few-shot 示例
        
        Args:
            examples: 检索到的样本列表
            
        Returns:
            格式化的 few-shot 字符串
        """
        fewshot_str = ""
        
        for i, ex in enumerate(examples, 1):
            # 格式化实体列表
            entities_json = json.dumps(ex['entities'], ensure_ascii=False, indent=2)
            
            fewshot_str += f"""
**示例 {i}** (相似度: {ex['similarity']:.3f})
Input: "{ex['text'][:500]}..."
Output:
```json
{entities_json}
```
"""
        
        return fewshot_str


def test_retriever():
    """测试检索器"""
    retriever = KNNRetriever(k=3)
    
    # 测试查询
    test_query = "P2-type layered Na0.67MnO2 cathode material with Fe doping shows improved cycling stability"
    
    print("\n" + "=" * 60)
    print("测试 KNN 检索")
    print("=" * 60)
    print(f"\n查询文本: {test_query[:100]}...")
    
    results = retriever.retrieve(test_query)
    
    print(f"\n检索到 {len(results)} 个相似样本:\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] 相似度: {r['similarity']:.4f}")
        print(f"    文本: {r['text'][:100]}...")
        print(f"    实体数: {len(r['entities'])}")
        if r['entities']:
            print(f"    实体示例: {r['entities'][:3]}")
        print()


if __name__ == "__main__":
    test_retriever()
