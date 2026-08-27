# -*- coding: utf-8 -*-
"""
直接使用 LLM 标注的基线配置。
使用 DeepSeek API 和 KNN few-shot 示例。
评估使用与监督基线一致的 tokenizer、BIO 标签和 seqeval F1。
"""

import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR     = os.path.join(PROJECT_DIR, "data", "annotation")
TEST_FILE    = os.path.join(DATA_DIR, "test_50.json")
TRAIN_FILE   = os.path.join(DATA_DIR, "train_30.json")  # 用于 KNN 检索的金标示例
OUTPUT_DIR   = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# DeepSeek API 配置
DEEPSEEK_API_KEY   = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL  = "https://api.deepseek.com"
DEEPSEEK_MODEL     = "deepseek-v4-flash"        

# KNN few-shot 配置
USE_KNN_FEWSHOT = True
KNN_K           = 3
KNN_MODEL       = "m3rg-iitd/matscibert"    # 检索编码器

# 评估 tokenizer
TOKENIZER_NAME = "m3rg-iitd/matscibert"
MAX_LENGTH     = 512

# 标签配置，与 baseline_65 保持一致
LABEL_LIST = [
    "O",
    "B-MATERIAL", "I-MATERIAL",
    "B-STRUCTURE", "I-STRUCTURE",
    "B-MODIFICATION", "I-MODIFICATION",
    "B-ROLE", "I-ROLE",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)
