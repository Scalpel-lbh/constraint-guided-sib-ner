# -*- coding: utf-8 -*-
"""
评估模块 - 计算NER指标
"""

from collections import defaultdict
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2


def convert_ids_to_labels(predictions, labels, id2label):
    """
    将ID转换回标签字符串，用于seqeval评估
    
    Args:
        predictions: list of list of int
        labels: list of list of int
        id2label: dict, ID到标签的映射
    
    Returns:
        pred_labels: list of list of str
        true_labels: list of list of str
    """
    pred_labels = []
    true_labels = []
    
    for pred_seq, label_seq in zip(predictions, labels):
        pred_tags = []
        true_tags = []
        
        for p, l in zip(pred_seq, label_seq):
            # 跳过padding（通常label=-100或特殊token）
            if l == -100:
                continue
            
            pred_tag = id2label.get(p, 'O')
            true_tag = id2label.get(l, 'O')
            
            pred_tags.append(pred_tag)
            true_tags.append(true_tag)
        
        pred_labels.append(pred_tags)
        true_labels.append(true_tags)
    
    return pred_labels, true_labels


def compute_metrics_from_predictions(predictions, labels, id2label):
    """
    从预测结果计算NER指标
    
    Args:
        predictions: list of list of int, 模型预测
        labels: list of list of int, 真实标签
        id2label: dict, ID到标签的映射
    
    Returns:
        dict: 包含各种指标的字典
    """
    pred_labels, true_labels = convert_ids_to_labels(predictions, labels, id2label)
    
    # 计算整体指标
    metrics = {
        'micro_f1': f1_score(true_labels, pred_labels, mode='strict', scheme=IOB2),
        'micro_precision': precision_score(true_labels, pred_labels, mode='strict', scheme=IOB2),
        'micro_recall': recall_score(true_labels, pred_labels, mode='strict', scheme=IOB2)
    }
    
    # 计算每个类别的指标
    report = classification_report(true_labels, pred_labels, mode='strict', scheme=IOB2, output_dict=True)
    
    # 提取各实体类型的指标
    for entity_type in ['MATERIAL', 'STRUCTURE', 'MODIFICATION', 'ROLE']:
        if entity_type in report:
            metrics[entity_type] = {
                'precision': float(report[entity_type]['precision']),
                'recall': float(report[entity_type]['recall']),
                'f1': float(report[entity_type]['f1-score']),
                'support': int(report[entity_type]['support'])
            }
        else:
            metrics[entity_type] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'support': 0
            }
    
    return metrics


def evaluate(model, dataloader, device, id2label):
    """
    评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        id2label: ID到标签的映射
    
    Returns:
        dict: 评估指标
    """
    import torch
    from tqdm import tqdm
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            # 获取预测（不带CRF，返回logits）
            logits = model(input_ids, attention_mask)
            
            # 从logits获取预测标签
            predictions = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)
            
            # 收集结果
            for pred, label, mask in zip(predictions, labels, attention_mask):
                valid_len = mask.sum().item()
                all_predictions.append(pred[:valid_len].cpu().tolist())
                all_labels.append(label[:valid_len].tolist())
    
    # 计算指标
    metrics = compute_metrics_from_predictions(all_predictions, all_labels, id2label)
    
    return metrics


def print_classification_report(predictions, labels, id2label):
    """打印详细的分类报告"""
    pred_labels, true_labels = convert_ids_to_labels(predictions, labels, id2label)
    
    print("\n" + "=" * 60)
    print("Classification Report:")
    print("=" * 60)
    print(classification_report(true_labels, pred_labels, mode='strict', scheme=IOB2))


if __name__ == '__main__':
    # 测试评估函数
    from config import ID2LABEL
    
    # 模拟数据
    predictions = [
        [0, 1, 2, 2, 0, 3, 4, 0],  # O, B-MAT, I-MAT, I-MAT, O, B-STR, I-STR, O
        [0, 0, 5, 6, 0, 7, 8, 0]   # O, O, B-MOD, I-MOD, O, B-ROLE, I-ROLE, O
    ]
    labels = [
        [0, 1, 2, 2, 0, 3, 4, 0],  # 完全正确
        [0, 0, 5, 6, 0, 0, 0, 0]   # 部分正确
    ]
    
    metrics = compute_metrics_from_predictions(predictions, labels, ID2LABEL)
    
    print("测试指标:")
    print(f"  Micro F1: {metrics['micro_f1']:.4f}")
    print(f"  Micro Precision: {metrics['micro_precision']:.4f}")
    print(f"  Micro Recall: {metrics['micro_recall']:.4f}")
