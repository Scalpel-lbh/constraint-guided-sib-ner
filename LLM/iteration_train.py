# -*- coding: utf-8 -*-
"""
主动学习迭代训练脚本

功能：
1. 将 LLM 标注数据转换为 Label Studio 格式
2. 与原始训练数据合并
3. 使用与基线相同的参数重新训练模型
4. 在测试集上评估
"""

import os
import sys
import json
import shutil
from datetime import datetime

# 路径配置（在导入其他模块前定义）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_DIR = os.path.join(BASE_DIR, "baseline_65")

# 添加 baseline_65 目录到路径最前面，确保优先导入
sys.path.insert(0, BASELINE_DIR)

from data_converter import convert_round_annotations, merge_with_training_data

LLM_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "annotation")

# 原始数据文件
ORIGINAL_TRAIN_FILE = os.path.join(DATA_DIR, "train_30.json")
VAL_FILE = os.path.join(DATA_DIR, "val_20.json")
TEST_FILE = os.path.join(DATA_DIR, "test_50.json")


def run_iteration(round_num):
    """
    运行一轮主动学习迭代训练
    
    Args:
        round_num: 轮次号
    """
    print("=" * 70)
    print(f"主动学习迭代训练 - Round {round_num}")
    print("=" * 70)
    
    # ========== Step 1: 转换 LLM 标注 ==========
    print("\n[Step 1] 转换 LLM 标注结果为 Label Studio 格式...")
    converted = convert_round_annotations(round_num)
    
    if not converted:
        print("错误: 转换失败")
        return None
    
    # ========== Step 2: 合并训练数据 ==========
    print("\n[Step 2] 合并训练数据...")
    
    # 创建本轮输出目录
    round_output_dir = os.path.join(LLM_DIR, "iterations", f"round_{round_num}")
    os.makedirs(round_output_dir, exist_ok=True)
    
    # 合并后的训练文件
    merged_train_file = os.path.join(round_output_dir, "train_merged.json")
    
    # 如果是第一轮，直接与原始训练数据合并
    # 如果是后续轮次，应该与上一轮的合并数据合并
    if round_num == 1:
        base_train_file = ORIGINAL_TRAIN_FILE
    else:
        prev_round_dir = os.path.join(LLM_DIR, "iterations", f"round_{round_num - 1}")
        base_train_file = os.path.join(prev_round_dir, "train_merged.json")
        if not os.path.exists(base_train_file):
            print(f"警告: 找不到上一轮训练数据，使用原始训练数据")
            base_train_file = ORIGINAL_TRAIN_FILE
    
    merged = merge_with_training_data(round_num, base_train_file, merged_train_file)
    
    # ========== Step 3: 重新训练模型 ==========
    print("\n[Step 3] 重新训练模型...")
    print("         使用与基线相同的参数")
    
    # 导入训练相关模块
    import random
    import numpy as np
    import torch
    from torch.optim import AdamW
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from tqdm import tqdm
    
    # 从 baseline_65 导入配置和模块
    from config import (
        MODEL_NAME, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, 
        LEARNING_RATE, CRF_LEARNING_RATE,
        NUM_EPOCHS, WARMUP_RATIO, WEIGHT_DECAY, MAX_GRAD_NORM,
        EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
        SEED, ID2LABEL, LABEL_LIST
    )
    from data_processor import create_dataloaders, save_bio_format  # type: ignore
    from model import MatSciBERTCRF  # type: ignore
    from evaluate import compute_metrics_from_predictions  # type: ignore
    from train import EarlyStopping, train_epoch, validate, set_seed  # type: ignore
    
    # 设置随机种子
    set_seed(SEED)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载 tokenizer
    print(f"加载 tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 导出 BIO 格式数据供检查
    print("导出数据检查文件...")
    bio_debug_file = os.path.join(round_output_dir, "train_merged_bio.txt")
    save_bio_format(merged_train_file, bio_debug_file, tokenizer)
    
    # 创建数据加载器（使用合并后的训练数据）
    print("加载数据...")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(
        merged_train_file, VAL_FILE, TEST_FILE, tokenizer,
        TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE
    )
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    # 创建模型
    print("创建模型...")
    model = MatSciBERTCRF()
    model = model.to(device)
    
    # 设置优化器（与基线相同的学习率配置）
    bert_params = []
    crf_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'crf' in name:
            crf_params.append(param)
        elif 'bert' in name:
            bert_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = AdamW([
        {'params': bert_params, 'lr': LEARNING_RATE},
        {'params': crf_params, 'lr': CRF_LEARNING_RATE},
        {'params': other_params, 'lr': LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY)
    
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
    
    # 训练循环
    print("\n开始训练...")
    best_model_path = os.path.join(round_output_dir, "best_model.pt")
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        history['train_loss'].append(train_loss)
        
        # 验证
        val_metrics = validate(model, val_loader, device, val_dataset, desc=f'Epoch {epoch} [Val]')
        val_loss = val_metrics['loss']
        val_f1 = val_metrics['micro_f1']
        
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}")
        
        # 早停检查
        if early_stopping(val_f1, epoch):
            print(f"  -> 保存最佳模型 (Val F1: {val_f1:.4f})")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
            }, best_model_path)
        
        if early_stopping.early_stop:
            print(f"\n早停触发! 最佳 epoch: {early_stopping.best_epoch}")
            break
    
    # ========== Step 4: 测试评估 ==========
    print("\n[Step 4] 在测试集上评估...")
    
    # 加载最佳模型
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试
    test_metrics = validate(model, test_loader, device, test_dataset, desc='Test')
    
    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)
    print(f"Micro F1:        {test_metrics['micro_f1']:.4f}")
    print(f"Micro Precision: {test_metrics['micro_precision']:.4f}")
    print(f"Micro Recall:    {test_metrics['micro_recall']:.4f}")
    print("\n各类别性能:")
    for label in ['MATERIAL', 'STRUCTURE', 'MODIFICATION', 'ROLE']:
        if label in test_metrics:
            m = test_metrics[label]
            print(f"  {label:15} F1={m['f1']:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}, Support={m['support']}")
    
    # ========== 保存结果 ==========
    results = {
        'round': round_num,
        'timestamp': datetime.now().isoformat(),
        'training_info': {
            'original_train_samples': len(json.load(open(ORIGINAL_TRAIN_FILE, 'r', encoding='utf-8'))),
            'llm_annotated_samples': len(converted),
            'total_train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'best_epoch': early_stopping.best_epoch,
            'best_val_f1': early_stopping.best_score
        },
        'test_metrics': test_metrics,
        'history': history
    }
    
    results_file = os.path.join(round_output_dir, "results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    
    print(f"\n结果保存至: {results_file}")
    
    return results


def compare_with_previous(round_results, round_num):
    """与上一轮(或基线)结果比较"""
    if round_num == 1:
        prev_results_file = os.path.join(BASELINE_DIR, "output", "results.json")
        prev_name = "基线 (30样本)"
    else:
        prev_results_file = os.path.join(LLM_DIR, "iterations", f"round_{round_num - 1}", "results.json")
        prev_name = f"Round {round_num - 1} ({30 + (round_num - 1) * 50}样本)"
        
    curr_name = f"Round {round_num} ({30 + round_num * 50}样本)"
    
    if not os.path.exists(prev_results_file):
        print(f"找不到对比结果文件: {prev_results_file}")
        return
    
    with open(prev_results_file, 'r', encoding='utf-8') as f:
        prev_data = json.load(f)
    
    prev_metrics = prev_data.get('test_metrics', {})
    round_metrics = round_results.get('test_metrics', {})
    
    print("\n" + "=" * 70)
    print(f"结果对比: {curr_name} vs {prev_name}")
    print("=" * 70)
    print(f"{'指标':<20} {prev_name:<25} {curr_name:<25} {'变化':<10}")
    print("-" * 70)
    
    for metric in ['micro_f1', 'micro_precision', 'micro_recall']:
        base_val = prev_metrics.get(metric, 0)
        round_val = round_metrics.get(metric, 0)
        diff = round_val - base_val
        diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        print(f"{metric:<20} {base_val:<25.4f} {round_val:<25.4f} {diff_str:<10}")
    
    print("-" * 70)
    print("\n各类别 F1 对比:")
    for label in ['MATERIAL', 'STRUCTURE', 'MODIFICATION', 'ROLE']:
        base_f1 = prev_metrics.get(label, {}).get('f1', 0)
        round_f1 = round_metrics.get(label, {}).get('f1', 0)
        diff = round_f1 - base_f1
        diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        print(f"  {label:<18} {base_f1:<25.4f} {round_f1:<25.4f} {diff_str:<10}")


if __name__ == '__main__':
    # 强制将标准输出重配置为 utf-8，彻底解决 Windows 下控制台打印乱码问题
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    round_num = 1
    if len(sys.argv) > 1:
        round_num = int(sys.argv[1])
    
    # 运行迭代训练
    results = run_iteration(round_num)
    
    if results:
        # 与上一轮(或基线)对比
        compare_with_previous(results, round_num)
