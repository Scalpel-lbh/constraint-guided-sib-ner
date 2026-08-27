# -*- coding: utf-8 -*-
"""
训练脚本 - MatSciBERT NER 基线实验
"""

import os
import json
import random
import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from datetime import datetime

from config import (
    MODEL_NAME, TRAIN_FILE, VAL_FILE, TEST_FILE, OUTPUT_DIR,
    TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, LEARNING_RATE, CRF_LEARNING_RATE,
    NUM_EPOCHS, WARMUP_RATIO, WEIGHT_DECAY, MAX_GRAD_NORM,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    SEED, DEVICE, ID2LABEL, LABEL_LIST
)
from data_processor import create_dataloaders
from model import MatSciBERTCRF
from evaluate import evaluate, compute_metrics_from_predictions


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return True  # 保存模型
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True  # 保存模型
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # 不保存模型


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        loss, logits = model(input_ids, attention_mask, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, device, dataset, desc='Val'):
    """验证/测试"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 计算损失和获取logits
            loss, logits = model(input_ids, attention_mask, labels)
            total_loss += loss.item()
            
            # 从logits获取预测标签
            predictions = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)
            
            # 收集预测和标签（只保留有效token）
            for i, (pred, label, mask) in enumerate(zip(predictions, labels, attention_mask)):
                valid_len = mask.sum().item()
                all_predictions.append(pred[:valid_len].cpu().tolist())
                all_labels.append(label[:valid_len].cpu().tolist())
    
    avg_loss = total_loss / len(dataloader)
    
    # 计算指标
    metrics = compute_metrics_from_predictions(all_predictions, all_labels, ID2LABEL)
    metrics['loss'] = avg_loss
    
    return metrics


def train(resume_from=None):
    """主训练函数"""
    # 设置随机种子
    set_seed(SEED)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设置设备
    device = DEVICE
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        device = 'cpu'
    print(f"使用设备: {device}")
    
    # 加载tokenizer
    print(f"\n加载tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 创建数据加载器
    print("\n加载数据...")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(
        TRAIN_FILE, VAL_FILE, TEST_FILE, tokenizer,
        TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE
    )
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    # 创建模型
    print("\n创建模型...")
    model = MatSciBERTCRF()
    model = model.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 设置优化器 - BERT和CRF使用不同学习率
    bert_params = []
    crf_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'bert' in name:
            bert_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = AdamW([
        {'params': bert_params, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY},
        {'params': other_params, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY}
    ])
    
    # 学习率调度器
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 早停
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        mode='max'
    )
    
    # 训练记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': []
    }
    
    print(f"\n开始训练...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {TRAIN_BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Warmup steps: {warmup_steps}")
    print("-" * 60)
    
    best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pt')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        
        # 验证
        val_metrics = validate(model, val_loader, device, val_dataset, desc=f'Epoch {epoch} [Val]')
        
        # 记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['micro_f1'])
        history['val_precision'].append(val_metrics['micro_precision'])
        history['val_recall'].append(val_metrics['micro_recall'])
        
        # 打印结果
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val F1: {val_metrics['micro_f1']:.4f} | P: {val_metrics['micro_precision']:.4f} | R: {val_metrics['micro_recall']:.4f}")
        
        # 早停检查
        should_save = early_stopping(val_metrics['micro_f1'], epoch)
        
        if should_save:
            print(f"  ✓ 保存最佳模型 (F1: {val_metrics['micro_f1']:.4f})")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1': val_metrics['micro_f1'],
                'history': history
            }, best_model_path)
        
        if early_stopping.early_stop:
            print(f"\n早停触发! 最佳epoch: {early_stopping.best_epoch}, 最佳F1: {early_stopping.best_score:.4f}")
            break
        
        print("-" * 60)
    
    # 加载最佳模型进行测试
    print("\n" + "=" * 60)
    print("加载最佳模型进行测试...")
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试
    test_metrics = validate(model, test_loader, device, test_dataset, desc='Test')
    
    print("\n" + "=" * 60)
    print("测试集结果:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Micro F1: {test_metrics['micro_f1']:.4f}")
    print(f"  Micro Precision: {test_metrics['micro_precision']:.4f}")
    print(f"  Micro Recall: {test_metrics['micro_recall']:.4f}")
    print("\n各类别详细结果:")
    for label in ['MATERIAL', 'STRUCTURE', 'MODIFICATION', 'ROLE']:
        if label in test_metrics:
            m = test_metrics[label]
            print(f"  {label}:")
            print(f"    F1: {m['f1']:.4f} | P: {m['precision']:.4f} | R: {m['recall']:.4f} | Support: {m['support']}")
    
    # 保存完整结果
    results = {
        'best_epoch': early_stopping.best_epoch,
        'best_val_f1': early_stopping.best_score,
        'test_metrics': test_metrics,
        'history': history,
        'config': {
            'model_name': MODEL_NAME,
            'train_batch_size': TRAIN_BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'crf_learning_rate': CRF_LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'seed': SEED
        }
    }
    
    results_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_path}")
    print("训练完成!")
    
    return model, results


if __name__ == '__main__':
    train()
