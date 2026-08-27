# -*- coding: utf-8 -*-
"""
评估模块
将字符偏移格式的预测结果转换为 BIO token 序列，
使用 seqeval 计算 F1 / Precision / Recall。
与 baseline_65 评估方式完全一致。
"""

import json
import os

from transformers import AutoTokenizer
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from config import (
    TOKENIZER_NAME, MAX_LENGTH,
    LABEL_LIST, OUTPUT_DIR,
)


def _load_tokenizer():
    print(f"加载 tokenizer: {TOKENIZER_NAME}")
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def char_spans_to_bio(text: str, entities: list, tokenizer, max_length: int) -> list:
    """
    将字符偏移实体列表转换为 BIO token 标签序列。

    Args:
        text:     原始摘要文本
        entities: [{"start": N, "end": M, "label": "MATERIAL"}, ...]
        tokenizer: HuggingFace tokenizer
        max_length: 最大序列长度

    Returns:
        bio_tags: list[str]，仅包含真实 token（去掉 [CLS]/[SEP] 及 padding）
    """
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_offsets_mapping=True,
        return_tensors='pt',
    )
    offset_mapping = encoding['offset_mapping'][0].tolist()

    # 初始化 O
    bio_tags = ['O'] * len(offset_mapping)

    # 按实体长度降序，防止短实体先占位覆盖长实体的 B- 标记
    sorted_ents = sorted(entities, key=lambda e: e['end'] - e['start'], reverse=True)

    for ent in sorted_ents:
        ent_start = ent['start']
        ent_end   = ent['end']
        label     = ent['label']
        is_first  = True
        for idx, (tok_s, tok_e) in enumerate(offset_mapping):
            if tok_s == 0 and tok_e == 0:   # [CLS] / [SEP] / PAD
                continue
            if tok_s < ent_end and tok_e > ent_start:
                if bio_tags[idx] == 'O':    # 未被占用才标
                    bio_tags[idx] = f'B-{label}' if is_first else f'I-{label}'
                    is_first = False
                else:
                    # 已有标签，仍推进 is_first 以保证后续 token 用 I-
                    if is_first:
                        is_first = False

    # 只保留真实 token（offset 非 (0,0) 且 attention_mask = 1）
    attention_mask = encoding['attention_mask'][0].tolist()
    filtered = [
        bio_tags[i]
        for i, (tok_s, tok_e) in enumerate(offset_mapping)
        if not (tok_s == 0 and tok_e == 0) and attention_mask[i] == 1
    ]
    return filtered


def evaluate(predictions_file: str = None, tokenizer=None):
    """
    读取 predictions.json，计算 seqeval 指标，并保存 results.json。
    """
    if predictions_file is None:
        predictions_file = os.path.join(OUTPUT_DIR, "predictions.json")

    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if tokenizer is None:
        tokenizer = _load_tokenizer()

    all_true, all_pred = [], []

    for item in data:
        abstract = item['abstract']
        true_bio = char_spans_to_bio(abstract, item['ground_truth'], tokenizer, MAX_LENGTH)
        pred_bio = char_spans_to_bio(abstract, item['prediction'],   tokenizer, MAX_LENGTH)

        # seqeval 要求两条序列等长
        min_len = min(len(true_bio), len(pred_bio))
        all_true.append(true_bio[:min_len])
        all_pred.append(pred_bio[:min_len])

    micro_f1  = f1_score(all_true, all_pred, average='micro')
    micro_p   = precision_score(all_true, all_pred, average='micro')
    micro_r   = recall_score(all_true, all_pred, average='micro')
    report    = classification_report(all_true, all_pred, output_dict=True)

    # ---- 打印 ----
    print("\n" + "=" * 60)
    print("LLM 直接预测基线（DeepSeek + KNN few-shot）测试结果")
    print("=" * 60)
    print(f"Micro F1:        {micro_f1:.4f}")
    print(f"Micro Precision: {micro_p:.4f}")
    print(f"Micro Recall:    {micro_r:.4f}")
    print("\n各类别性能:")
    entity_types = ['MATERIAL', 'STRUCTURE', 'MODIFICATION', 'ROLE']
    support_map = {}
    for etype in entity_types:
        row = report.get(etype, {})
        support_map[etype] = int(row.get('support', 0))
        print(f"  {etype:<14}  F1={row.get('f1-score',0):.4f}, "
              f"P={row.get('precision',0):.4f}, "
              f"R={row.get('recall',0):.4f}, "
              f"Support={support_map[etype]}")

    # ---- 保存 ----
    results = {
        "total_samples":    len(data),
        "model":            "deepseek-v4-flash (direct, no training)",
        "knn_fewshot":      True,
        "test_metrics": {
            "micro_f1":        micro_f1,
            "micro_precision": micro_p,
            "micro_recall":    micro_r,
        }
    }
    for etype in entity_types:
        row = report.get(etype, {})
        results["test_metrics"][etype] = {
            "precision": row.get('precision', 0),
            "recall":    row.get('recall', 0),
            "f1":        row.get('f1-score', 0),
            "support":   support_map[etype],
        }

    out_file = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至: {out_file}")

    return results
