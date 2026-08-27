# -*- coding: utf-8 -*-
"""
数据预处理模块 - 将Label Studio格式转换为NER训练格式
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import LABEL2ID, MAX_LENGTH, MODEL_NAME


def load_labelstudio_data(filepath):
    """加载Label Studio导出的JSON数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def convert_labelstudio_to_ner(record):
    """
    将Label Studio单条记录转换为NER格式
    
    返回:
        text: 原始文本
        entities: [(start, end, label), ...] 实体列表
    """
    text = record['data']['text']
    entities = []
    
    if record.get('annotations'):
        for anno in record['annotations']:
            for result in anno.get('result', []):
                if result.get('type') == 'labels':
                    value = result.get('value', {})
                    start = value.get('start')
                    end = value.get('end')
                    labels = value.get('labels', [])
                    if labels and start is not None and end is not None:
                        entities.append((start, end, labels[0]))
    
    # 按起始位置排序
    entities.sort(key=lambda x: x[0])
    return text, entities


def align_labels_with_tokens(text, entities, tokenizer, max_length):
    """
    将字符级别的实体标注对齐到token级别
    
    使用BIO标注格式
    """
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,   #超过最大token长度就截断
        padding='max_length',
        return_offsets_mapping=True,
        return_tensors='pt'
    )
    
    # 获取offset mapping (每个token对应原文的字符范围)
    offset_mapping = encoding['offset_mapping'][0].tolist()
    
    # 初始化所有标签为'O'
    labels = ['O'] * len(offset_mapping)
    
    # 为每个实体分配标签
    for ent_start, ent_end, ent_label in entities:
        # 实体可能被截断
        if ent_start >= len(text):
            continue
            
        is_first_token = True
        for idx, (token_start, token_end) in enumerate(offset_mapping):
            # 跳过特殊token ([CLS], [SEP], [PAD])
            if token_start == 0 and token_end == 0:
                continue
            
            # 检查token是否与实体重叠
            if token_start < ent_end and token_end > ent_start:
                if is_first_token:
                    labels[idx] = f'B-{ent_label}'
                    is_first_token = False
                else:
                    labels[idx] = f'I-{ent_label}'
    
    # 转换为ID
    label_ids = [LABEL2ID.get(label, 0) for label in labels]
    
    return {
        'input_ids': encoding['input_ids'].squeeze(0),
        'attention_mask': encoding['attention_mask'].squeeze(0),
        'labels': torch.tensor(label_ids, dtype=torch.long),
        'offset_mapping': offset_mapping
    }


def save_bio_format(input_file, output_file, tokenizer, max_length=MAX_LENGTH):
    """
    调试工具：将 Label Studio 数据转换为可视化的 BIO 文本文件
    用于检查数据对齐和实体转换是否正确
    """
    print(f"正在导出 BIO 检查文件: {output_file}")
    data = load_labelstudio_data(input_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, record in enumerate(data):
            text, entities = convert_labelstudio_to_ner(record)
            
            # 使用与训练完全相同的 Tokenizer 参数
            encoding = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_offsets_mapping=True
            )
            
            offset_mapping = encoding['offset_mapping']
            input_ids = encoding['input_ids']
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            # 复用转换逻辑生成 BIO 标签
            labels = ['O'] * len(offset_mapping)
            
            for ent_start, ent_end, ent_label in entities:
                if ent_start >= len(text):
                    continue
                    
                is_first_token = True
                for i, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start == 0 and token_end == 0:
                        continue
                        
                    if token_start < ent_end and token_end > ent_start:
                        if is_first_token:
                            labels[i] = f'B-{ent_label}'
                            is_first_token = False
                        else:
                            labels[i] = f'I-{ent_label}'
            
            # 写入文件
            f.write(f"Sample {idx} (ID: {record.get('id', 'N/A')})\n")
            f.write(f"Original Text: {text[:200]}...\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Token':<20} {'Label':<20}\n")
            f.write("-" * 60 + "\n")
            
            # 只输出非 padding 的部分
            for token, label in zip(tokens, labels):
                if token in ['[PAD]']:
                    continue
                f.write(f"{token:<20} {label}\n")
                
            f.write("=" * 80 + "\n\n")
    
    print(f"BIO 数据已导出至: {output_file}")



class NERDataset(Dataset):
    """NER数据集类"""
    
    def __init__(self, filepath, tokenizer, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # 加载并转换数据
        raw_data = load_labelstudio_data(filepath)
        
        for record in raw_data:
            text, entities = convert_labelstudio_to_ner(record)
            sample = align_labels_with_tokens(text, entities, tokenizer, max_length)
            sample['text'] = text
            sample['entities'] = entities
            self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'input_ids': sample['input_ids'],
            'attention_mask': sample['attention_mask'],
            'labels': sample['labels']
        }


def create_dataloaders(train_file, val_file, test_file, tokenizer, 
                       train_batch_size, eval_batch_size):
    """创建数据加载器"""
    from torch.utils.data import DataLoader
    
    train_dataset = NERDataset(train_file, tokenizer)
    val_dataset = NERDataset(val_file, tokenizer)
    test_dataset = NERDataset(test_file, tokenizer)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=eval_batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=eval_batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    # 测试数据加载
    from config import TRAIN_FILE, VAL_FILE, TEST_FILE
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("加载训练集...")
    train_dataset = NERDataset(TRAIN_FILE, tokenizer)
    print(f"训练集样本数: {len(train_dataset)}")
    
    print("\n加载验证集...")
    val_dataset = NERDataset(VAL_FILE, tokenizer)
    print(f"验证集样本数: {len(val_dataset)}")
    
    print("\n加载测试集...")
    test_dataset = NERDataset(TEST_FILE, tokenizer)
    print(f"测试集样本数: {len(test_dataset)}")
    
    # 查看一个样本
    print("\n=== 样本示例 ===")
    sample = train_dataset[0]
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"attention_mask shape: {sample['attention_mask'].shape}")
    print(f"labels shape: {sample['labels'].shape}")
