# -*- coding: utf-8 -*-
"""
Transformer-CRF 命名实体识别基线配置。
"""

import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'annotation')
OUTPUT_DIR = os.environ.get('MATSCIBERT_OUTPUT_DIR', os.path.join(BASE_DIR, 'output'))

# 数据集文件
TRAIN_FILE = os.path.join(DATA_DIR, 'train_30.json')
VAL_FILE = os.path.join(DATA_DIR, 'val_20.json')
TEST_FILE = os.path.join(DATA_DIR, 'test_50.json')

# 模型配置
MODEL_NAME = "m3rg-iitd/matscibert"  # MatSciBERT 编码器
MAX_LENGTH = 512  # BERT 最大序列长度
DROPOUT_RATE = 0.2  # Dropout 比例

# BIO 标签配置
LABEL_LIST = [
    'O',           # 非实体
    'B-MATERIAL',  # 材料实体起始
    'I-MATERIAL',  # 材料实体内部
    'B-STRUCTURE', # 结构实体起始
    'I-STRUCTURE', # 结构实体内部
    'B-MODIFICATION', # 改性实体起始
    'I-MODIFICATION', # 改性实体内部
    'B-ROLE',      # 角色实体起始
    'I-ROLE',      # 角色实体内部
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

# 训练参数
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 1e-5
CRF_LEARNING_RATE = 5e-4
NUM_EPOCHS = 35
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# 早停配置
EARLY_STOPPING_PATIENCE = 7  # 连续七轮无提升时停止
EARLY_STOPPING_MIN_DELTA = 0.001  # 验证指标最小提升量

# 随机种子
SEED = int(os.environ.get('MATSCIBERT_SEED', 42))

# 运行设备
DEVICE = "cuda"  # 使用 CUDA 设备训练
