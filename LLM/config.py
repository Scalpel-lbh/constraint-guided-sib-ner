# -*- coding: utf-8 -*-
"""
迭代式 LLM 标注配置。
"""

import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
BASELINE_DIR = os.path.join(BASE_DIR, "baseline_65")
AL_DIR = os.path.dirname(os.path.abspath(__file__))

# 30 篇人工金标训练集，用于 KNN 和 few-shot 示例
ORIGINAL_TRAIN_FILE = os.path.join(DATA_DIR, "annotation", "train_30.json")

# 未标注样本池
POOL_FILE = os.path.join(DATA_DIR, "Na", "remain_2581.json")

# 已选择样本记录
SELECTED_SAMPLES_FILE = os.path.join(AL_DIR, "selected_samples.json")

# 各轮采样输出目录
SAMPLE_OUTPUT_DIR = os.path.join(AL_DIR, "sampled")

# 监督基线模型和编码器
MODEL_PATH = os.path.join(BASELINE_DIR, "output", "best_model.pt")
MODEL_NAME = "bert-base-cased"

# 采样配置
# 每轮选择 25 篇结构类文档和 25 篇改性类文档
STRUCTURE_SAMPLE_NUM = 25
MODIFICATION_SAMPLE_NUM = 25
UNCERTAINTY_SAMPLE_NUM = 0  # 不使用不确定性采样
TOTAL_SAMPLE_NUM = STRUCTURE_SAMPLE_NUM + MODIFICATION_SAMPLE_NUM  # 50

# 结构类采样关键词
STRUCTURE_KEYWORDS = [
    "layered", "spinel", "olivine", "NASICON", "NASICON-type",
    "Prussian blue framework", "P2", "O3", "P3", "O2",
    "tunnel", "tunnel structure"
]

# 改性类采样关键词
MODIFICATION_KEYWORDS = [
    "doped", "doping", "co-doped", "substituted", "substitution",
    "oxygen-vacancy", "defect-rich", "coated", "coating",
    "carbon-coated", "surface-modified"
]

# 编码器配置
MAX_LENGTH = 512
DEVICE = "cuda"

# 标签配置，与监督基线保持一致
LABEL_LIST = [
    "O",
    "B-MATERIAL", "I-MATERIAL",
    "B-STRUCTURE", "I-STRUCTURE", 
    "B-MODIFICATION", "I-MODIFICATION",
    "B-ROLE", "I-ROLE"
]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)
