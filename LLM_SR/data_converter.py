# -*- coding: utf-8 -*-
"""
数据转换器

将 LLM 标注结果转换为 Label Studio 格式，与原始训练数据合并
复用 LLM 的核心逻辑
"""

import json
import os
import sys
import uuid
import re
import importlib.util

# 添加父目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ANNOTATED_OUTPUT_DIR, ITERATIONS_DIR,
    ORIGINAL_TRAIN_FILE, LABEL_LIST
)

# 复用 LLM 的数据转换核心逻辑
from LLM.data_converter import align_predictions


VALID_ENTITY_LABELS = {
    label.split('-', 1)[1]
    for label in LABEL_LIST
    if label.startswith('B-')
}


def _load_baseline_modules(baseline_dir: str):
    """Load baseline_65 config/data_processor without leaking module name conflicts."""
    config_path = os.path.join(baseline_dir, 'config.py')
    processor_path = os.path.join(baseline_dir, 'data_processor.py')

    config_spec = importlib.util.spec_from_file_location('baseline65_config', config_path)
    processor_spec = importlib.util.spec_from_file_location('baseline65_data_processor', processor_path)
    if config_spec is None or config_spec.loader is None:
        raise ImportError(f'无法加载 baseline config: {config_path}')
    if processor_spec is None or processor_spec.loader is None:
        raise ImportError(f'无法加载 baseline data_processor: {processor_path}')

    baseline_config = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(baseline_config)

    # data_processor.py uses "from config import ...".
    # Temporarily map "config" to baseline config so it never picks LLM_SR/config.py.
    original_config_module = sys.modules.get('config')
    data_processor = importlib.util.module_from_spec(processor_spec)
    try:
        sys.modules['config'] = baseline_config
        processor_spec.loader.exec_module(data_processor)
    finally:
        if original_config_module is not None:
            sys.modules['config'] = original_config_module
        else:
            sys.modules.pop('config', None)

    return baseline_config, data_processor


def convert_round_annotations(round_num: int):
    """
    转换指定轮次的 LLM 标注结果为 Label Studio 格式
    
    Args:
        round_num: 轮次号
        
    Returns:
        converted: 转换后的 Label Studio 格式数据列表，失败返回 None
    """
    # 加载 LLM 标注结果
    annotated_file = os.path.join(ANNOTATED_OUTPUT_DIR, f"round_{round_num}_annotated.json")
    
    if not os.path.exists(annotated_file):
        print(f"错误: 找不到标注文件 {annotated_file}")
        return None
    
    with open(annotated_file, 'r', encoding='utf-8') as f:
        annotated_data = json.load(f)
    
    results = annotated_data['results']
    print(f"加载 {len(results)} 个 LLM 标注结果（带结构纠错）")
    
    # 转换为 Label Studio 格式
    converted = []
    total_entities = 0
    aligned_entities = 0
    
    for result in results:
        # 跳过失败的样本（与对照组一致）
        if not result.get('success', False) or result.get('entities') is None:
            print(f"  跳过失败样本: {result.get('title', '')[:50]}")
            continue
            
        abstract = result.get('abstract', '')
        title = result.get('title', '')
        
        # 重要：必须使用 entities 字段（已对齐的实体），而不是 entities_raw
        # entities_raw 是 LLM 原始输出，可能包含无法对齐到文本的实体
        # entities 是经过 align_predictions 对齐后的结果，位置准确
        entities = result.get('entities', [])
        # Drop non-NER labels (e.g., "O") before alignment to avoid span conflicts.
        entities = [
            e for e in entities
            if isinstance(e, dict)
            and (e.get('label') or '').strip().upper() in VALID_ENTITY_LABELS
            and (e.get('text') or '').strip()
            and (e.get('context') or '').strip()
        ]
        
        if not entities:
            # 没有实体的样本也要保留
            converted.append({
                'id': str(uuid.uuid4()),
                'data': {
                    'text': abstract,
                    'title': title
                },
                'annotations': [{
                    'result': []
                }]
            })
            continue
        
        # 使用 align_predictions 对齐位置（参数顺序：原文, 实体列表）
        # align_predictions 直接返回 Label Studio 格式的结果
        ls_results = align_predictions(abstract, entities)
        
        total_entities += len(entities)
        aligned_entities += len(ls_results)
        
        converted.append({
            'id': str(uuid.uuid4()),
            'data': {
                'text': abstract,
                'title': title
            },
            'annotations': [{
                'result': ls_results
            }]
        })
    
    # 保存转换结果
    output_file = os.path.join(ANNOTATED_OUTPUT_DIR, f"round_{round_num}_labelstudio.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    
    print(f"成功转换 {len(converted)} 个样本")
    print(f"实体对齐率: {aligned_entities}/{total_entities} ({aligned_entities/total_entities*100:.1f}%)" if total_entities > 0 else "无实体")
    print(f"保存至: {output_file}")
    
    # 生成 BIO 格式检查文件（新增数据的BIO形式）
    print(f"\n生成新增数据的BIO格式文件...")
    
    # 动态加载 baseline_65 模块，避免与 LLM_SR/config.py 同名冲突
    baseline_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baseline_65")
    baseline_config, data_processor = _load_baseline_modules(baseline_dir)

    from transformers import AutoTokenizer

    bio_output_file = os.path.join(ANNOTATED_OUTPUT_DIR, f"round_{round_num}_bio.txt")
    tokenizer = AutoTokenizer.from_pretrained(baseline_config.MODEL_NAME)
    data_processor.save_bio_format(output_file, bio_output_file, tokenizer)
    print(f"BIO格式已保存至: {bio_output_file}")
    
    return converted


def merge_with_training_data(round_num: int, base_train_file: str, output_file: str) -> bool:
    """
    将 LLM 标注数据与训练数据合并
    """
    # 加载基础训练数据
    with open(base_train_file, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    
    # 加载转换后的 LLM 标注
    ls_file = os.path.join(ANNOTATED_OUTPUT_DIR, f"round_{round_num}_labelstudio.json")
    with open(ls_file, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)
    
    # 合并
    merged = base_data + llm_data
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    print(f"原始训练数据: {len(base_data)} 样本")
    print(f"LLM 标注数据: {len(llm_data)} 样本")
    print(f"合并后: {len(merged)} 样本")
    print(f"保存至: {output_file}")
    
    return True


if __name__ == '__main__':
    import sys
    
    round_num = 1
    if len(sys.argv) > 1:
        round_num = int(sys.argv[1])
    
    print(f"转换 Round {round_num} 标注数据...")
    convert_round_annotations(round_num)
