# -*- coding: utf-8 -*-
"""
约束引导的 LLM 标注配置。
包括提示词、KNN 示例、错误修正和遗漏实体恢复。
"""

import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
BASELINE_DIR = os.path.join(BASE_DIR, "baseline_65")
AL_DIR = os.path.dirname(os.path.abspath(__file__))  # AL_LLM_SR

# 基础 LLM 标注流程目录
AL_BASELINE_DIR = os.path.join(BASE_DIR, "LLM")

# 30 篇人工金标训练集，用于 KNN 和 few-shot 示例
ORIGINAL_TRAIN_FILE = os.path.join(DATA_DIR, "annotation", "train_30.json")

# 复用基础 LLM 标注流程的输入
# 复用基础 LLM 流程逐轮采样的文档
SAMPLE_INPUT_DIR = os.path.join(AL_BASELINE_DIR, "sampled")
# 保留基础 LLM 标注，供比较和约束修正使用
BASELINE_ANNOTATED_DIR = os.path.join(AL_BASELINE_DIR, "annotated")

# 约束引导标注输出
# 修正后的标注与基础标注分开保存
ANNOTATED_OUTPUT_DIR = os.path.join(AL_DIR, "annotated")

# 各轮累积训练数据输出目录
ITERATIONS_DIR = os.path.join(AL_DIR, "iterations")

# 编码器配置
MODEL_NAME = "bert-base-cased"
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

# LLM API 配置
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-v4-flash"

# KNN few-shot 配置
USE_KNN_FEWSHOT = True
KNN_K = 3  # 最近邻示例数量

# 约束修正配置
MAX_REFINEMENT_ROUNDS = 1  # 最大修正轮数

# 下游 NER 训练配置
# 参数与对应的监督训练配置保持一致
# iteration_train.py 使用累积银标数据进行训练
TRAINING_PARAM_OVERRIDES = {
    'LEARNING_RATE': 1e-5,
    'DROPOUT_RATE': 0.2,
    'WARMUP_RATIO': 0.05,
    'EARLY_STOPPING_PATIENCE': 7,
    'NUM_EPOCHS': 35,
}

# 错误分析输出路径
os.makedirs(ANNOTATED_OUTPUT_DIR, exist_ok=True)
os.makedirs(ITERATIONS_DIR, exist_ok=True)
