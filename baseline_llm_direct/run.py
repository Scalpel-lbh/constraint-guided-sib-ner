# -*- coding: utf-8 -*-
"""
LLM 直接预测基线 主入口

用法:
  python run.py          -- 预测 + 评估（支持断点续跑）
  python run.py --eval   -- 仅评估（已有 predictions.json 时跳过预测）
"""

import argparse
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 确保当前目录优先，LLM 追加到末尾（避免 config 被覆盖）
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
LLM_DIR  = os.path.join(PROJECT_DIR, "LLM")
if LLM_DIR not in sys.path:
    sys.path.append(LLM_DIR)         # 低优先级
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)        # 最高优先级

from config import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true',
                        help='仅评估（跳过预测，直接读取已有 predictions.json）')
    args = parser.parse_args()

    predictions_file = os.path.join(OUTPUT_DIR, "predictions.json")

    # ---------- Step 1: 预测 ----------
    if not args.eval:
        print("=" * 60)
        print("Step 1  DeepSeek + KNN few-shot 预测测试集")
        print("=" * 60)
        from predictor import predict_test_set
        predict_test_set(resume=True)
    else:
        if not os.path.exists(predictions_file):
            print(f"错误: {predictions_file} 不存在，请先运行预测。")
            sys.exit(1)
        print(f"[--eval] 跳过预测，直接读取 {predictions_file}")

    # ---------- Step 2: 评估 ----------
    print("\n" + "=" * 60)
    print("Step 2  评估（字符偏移 → BIO token → seqeval）")
    print("=" * 60)
    from evaluator import evaluate
    evaluate(predictions_file)


if __name__ == "__main__":
    main()
