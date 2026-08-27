# -*- coding: utf-8 -*-
"""
配置文件 - NER 基线实验
"""

import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'annotation')
OUTPUT_DIR = os.environ.get('BERT_OUTPUT_DIR', os.path.join(BASE_DIR, 'output'))

# 数据文件
TRAIN_FILE = os.path.join(DATA_DIR, 'train_30.json')
VAL_FILE = os.path.join(DATA_DIR, 'val_20.json')
TEST_FILE = os.path.join(DATA_DIR, 'test_50.json')

# 模型配置
MODEL_NAME = "bert-base-cased"  #
MAX_LENGTH = 512  # BERT最大序列长度
DROPOUT_RATE = 0.2  

# 标签配置 (BIO标注格式)
LABEL_LIST = [
    'O',           # 非实体
    'B-MATERIAL',  # 材料开始
    'I-MATERIAL',  # 材料内部
    'B-STRUCTURE', # 结构开始
    'I-STRUCTURE', # 结构内部
    'B-MODIFICATION', # 修饰开始
    'I-MODIFICATION', # 修饰内部
    'B-ROLE',      # 角色开始
    'I-ROLE',      # 角色内部
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

# 训练参数
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 1e-5  # 降低学习率
CRF_LEARNING_RATE = 5e-4  # CRF层使用更大的学习率
NUM_EPOCHS = 35
WARMUP_RATIO = 0.05  # 减小warmup
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# 早停配置
EARLY_STOPPING_PATIENCE = 7  # 验证集F1连续7个epoch不提升则停止
EARLY_STOPPING_MIN_DELTA = 0.001  # 最小提升阈值

# 随机种子
SEED = int(os.environ.get('BERT_SEED', 42))

# 设备
DEVICE = "cuda"  # 如果没有GPU会自动降级到CPU
