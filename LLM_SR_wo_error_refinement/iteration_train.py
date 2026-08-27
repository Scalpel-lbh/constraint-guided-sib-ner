# -*- coding: utf-8 -*-
"""
主动学习迭代训练脚本（实验组）

功能：
1. 将 LLM 标注数据（带结构纠错）转换为 Label Studio 格式
2. 与原始训练数据合并
3. 使用与基线相同的参数重新训练模型
4. 在测试集上评估

注意：训练逻辑与对照组 (LLM/iteration_train.py) 完全一致
"""

import os
import sys
import json
import shutil
from datetime import datetime

# 路径配置（在导入其他模块前定义）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_DIR = os.path.join(BASE_DIR, "baseline_65")
LLM_DIR = os.path.dirname(os.path.abspath(__file__))

# 首先导入本目录的模块（LLM_SR 的 config 和 data_converter）
from config import ORIGINAL_TRAIN_FILE, ITERATIONS_DIR, TRAINING_PARAM_OVERRIDES
from data_converter import convert_round_annotations, merge_with_training_data

# 然后添加 baseline_65 到路径，以便导入训练相关模块
sys.path.insert(0, BASELINE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data", "annotation")

# 原始数据文件（与对照组一致）
VAL_FILE = os.path.join(DATA_DIR, "val_20.json")
TEST_FILE = os.path.join(DATA_DIR, "test_50.json")


def run_iteration(round_num):
    """
    运行一轮主动学习迭代训练
    
    Args:
        round_num: 轮次号
    """
    print("=" * 70)
    print(f"主动学习迭代训练（实验组：结构纠错）- Round {round_num}")
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
    round_output_dir = os.path.join(ITERATIONS_DIR, f"round_{round_num}")
    os.makedirs(round_output_dir, exist_ok=True)
    
    # 合并后的训练文件
    merged_train_file = os.path.join(round_output_dir, "train_merged.json")
    
    # 如果是第一轮，直接与原始训练数据合并
    # 如果是后续轮次，应该与上一轮的合并数据合并
    if round_num == 1:
        base_train_file = ORIGINAL_TRAIN_FILE
    else:
        prev_round_dir = os.path.join(ITERATIONS_DIR, f"round_{round_num - 1}")
        base_train_file = os.path.join(prev_round_dir, "train_merged.json")
        if not os.path.exists(base_train_file):
            print(f"警告: 找不到上一轮训练数据，使用原始训练数据")
            base_train_file = ORIGINAL_TRAIN_FILE
    
    merge_with_training_data(round_num, base_train_file, merged_train_file)
    
    # ========== Step 3: 重新训练模型 ==========
    print("\n[Step 3] 重新训练模型...")
    print("         使用与基线相同的参数")
    
    # 导入训练相关模块（与对照组完全一致）
    import random
    import numpy as np
    import torch
    from torch.optim import AdamW
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from tqdm import tqdm
    
    # 重要：为了避免模块缓存冲突，需要临时修改环境
    # 1. 保存当前工作目录
    original_cwd = os.getcwd()
    # 2. 保存原有 sys.path
    original_path = sys.path.copy()
    
    # 3. 清除可能已缓存的 baseline_65 相关模块
    modules_to_remove = [k for k in sys.modules.keys() 
                         if k in ('config', 'train', 'model', 'evaluate', 'data_processor')
                         or k.startswith('config.') or k.startswith('train.')]
    for mod in modules_to_remove:
        del sys.modules[mod]
    
    # 4. 切换到 baseline_65 目录并调整路径
    os.chdir(BASELINE_DIR)
    sys.path = [BASELINE_DIR] + [p for p in sys.path if 'LLM_SR' not in p]
    
    # 5. 现在可以安全导入 baseline_65 的模块
    from config import (
        MODEL_NAME, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, 
        LEARNING_RATE, CRF_LEARNING_RATE,
        NUM_EPOCHS, WARMUP_RATIO, WEIGHT_DECAY, MAX_GRAD_NORM,
        EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
        SEED, ID2LABEL, LABEL_LIST, DROPOUT_RATE
    )
    from data_processor import create_dataloaders, save_bio_format  # type: ignore
    from model import MatSciBERTCRF  # type: ignore
    from evaluate import compute_metrics_from_predictions  # type: ignore
    from train import EarlyStopping, train_epoch, validate, set_seed  # type: ignore
    
    # 6. 恢复原工作目录（但保留 baseline_65 在 path 中以便后续使用）
    os.chdir(original_cwd)

    # 应用 LLM_SR 组的训练参数覆盖（不影响 LLM / baseline）
    overrides = TRAINING_PARAM_OVERRIDES or {}
    LEARNING_RATE = overrides.get('LEARNING_RATE', LEARNING_RATE)
    CRF_LEARNING_RATE = overrides.get('CRF_LEARNING_RATE', CRF_LEARNING_RATE)
    DROPOUT_RATE = overrides.get('DROPOUT_RATE', DROPOUT_RATE)
    NUM_EPOCHS = overrides.get('NUM_EPOCHS', NUM_EPOCHS)
    WARMUP_RATIO = overrides.get('WARMUP_RATIO', WARMUP_RATIO)
    WEIGHT_DECAY = overrides.get('WEIGHT_DECAY', WEIGHT_DECAY)
    EARLY_STOPPING_PATIENCE = overrides.get('EARLY_STOPPING_PATIENCE', EARLY_STOPPING_PATIENCE)
    EARLY_STOPPING_MIN_DELTA = overrides.get('EARLY_STOPPING_MIN_DELTA', EARLY_STOPPING_MIN_DELTA)
    SEED = overrides.get('SEED', SEED)
    
    # 设置随机种子
    set_seed(SEED)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print("训练参数:")
    print(f"  LEARNING_RATE={LEARNING_RATE}")
    print(f"  DROPOUT_RATE={DROPOUT_RATE}")
    print(f"  WARMUP_RATIO={WARMUP_RATIO}")
    print(f"  NUM_EPOCHS={NUM_EPOCHS}")
    print(f"  EARLY_STOPPING_PATIENCE={EARLY_STOPPING_PATIENCE}")
    print(f"  SEED={SEED}")
    
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
    model = MatSciBERTCRF(dropout_rate=DROPOUT_RATE)
    model = model.to(device)
    
    # 设置优化器：默认与 baseline 一致；仅在模型确有 CRF 参数时启用单独 CRF 学习率
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
    
    if crf_params:
        print(f"  检测到 CRF 参数组: {len(crf_params)}，CRF_LEARNING_RATE={CRF_LEARNING_RATE}")
        optimizer = AdamW([
            {'params': bert_params, 'lr': LEARNING_RATE},
            {'params': crf_params, 'lr': CRF_LEARNING_RATE},
            {'params': other_params, 'lr': LEARNING_RATE}
        ], weight_decay=WEIGHT_DECAY)
    else:
        print("  未检测到 CRF 参数组，使用无 CRF 优化配置")
        optimizer = AdamW([
            {'params': bert_params, 'lr': LEARNING_RATE},
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
        'method': 'Prompt + KNN + Missing-Entity Recovery Only',
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


def compare_with_baseline(round_results):
    """与基线结果比较"""
    baseline_results_file = os.path.join(BASELINE_DIR, "output", "results.json")
    
    if not os.path.exists(baseline_results_file):
        print("找不到基线结果文件")
        return
    
    with open(baseline_results_file, 'r', encoding='utf-8') as f:
        baseline = json.load(f)
    
    baseline_metrics = baseline.get('test_metrics', {})
    round_metrics = round_results.get('test_metrics', {})
    
    print("\n" + "=" * 70)
    print("与基线对比")
    print("=" * 70)
    print(f"{'指标':<20} {'基线 (30样本)':<15} {'实验组 Round':<15} {'变化':<10}")
    print("-" * 70)
    
    for metric in ['micro_f1', 'micro_precision', 'micro_recall']:
        base_val = baseline_metrics.get(metric, 0)
        round_val = round_metrics.get(metric, 0)
        diff = round_val - base_val
        diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        print(f"{metric:<20} {base_val:<15.4f} {round_val:<15.4f} {diff_str:<10}")
    
    print("-" * 70)
    print("\n各类别 F1 对比:")
    for label in ['MATERIAL', 'STRUCTURE', 'MODIFICATION', 'ROLE']:
        base_f1 = baseline_metrics.get(label, {}).get('f1', 0)
        round_f1 = round_metrics.get(label, {}).get('f1', 0)
        diff = round_f1 - base_f1
        diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        print(f"  {label:<15} {base_f1:<15.4f} {round_f1:<15.4f} {diff_str:<10}")


if __name__ == '__main__':
    round_num = 1
    if len(sys.argv) > 1:
        round_num = int(sys.argv[1])
    
    # 运行迭代训练
    results = run_iteration(round_num)
    
    if results:
        # 与基线对比
        compare_with_baseline(results)
