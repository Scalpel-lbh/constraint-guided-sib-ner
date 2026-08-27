# -*- coding: utf-8 -*-
"""
数据转换模块 - 将 LLM 标注结果转换为 Label Studio 格式
"""

import json
import os
import re
import uuid
from datetime import datetime

# 直接定义路径，避免从 config 导入
LLM_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_OUTPUT_DIR = os.path.join(LLM_DIR, "sampled")

# 标注输出目录
ANNOTATION_OUTPUT_DIR = os.path.join(LLM_DIR, "annotated")


def generate_ls_id():
    """生成 Label Studio 需要的随机 ID"""
    return str(uuid.uuid4())[:10]


def align_predictions(original_text, llm_entities):
    """
    基于上下文锚点的精准对齐算法（改进版：两步定位策略）
    
    结合了：
    - 长词优先（解决嵌套问题，如 "Carbon coated" vs "Carbon"）
    - 上下文定位（解决多义/重复问题，如区分两个 "P2"）
    - 掩码占位（解决重叠问题）
    - 单词边界检查（防止 "Na" 匹配到 "Nature"）
    - 【新增】两步定位：先context精确定位，再fallback到全文搜索
    
    Args:
        original_text: 原始摘要文本
        llm_entities: LLM 标注的实体列表 [{"text": ..., "label": ..., "context": ...}, ...]
        
    Returns:
        results: Label Studio 格式的标注结果列表
    """
    if not llm_entities:
        return []
    
    # 1. 关键步骤：按实体长度降序排序
    # 解决 "Carbon coated" vs "Carbon" 问题：先标长的，占位后，短的就标不进去了
    sorted_entities = sorted(llm_entities, key=lambda x: len(x.get('text', '')), reverse=True)
    
    # 2. 初始化占用掩码 (Mask)
    # 0 表示该字符位置空闲，1 表示已被占用
    char_mask = [0] * len(original_text)
    
    results = []
    failed_entities = []  # 第一轮失败的实体，留给第二轮
    
    # ========== 第一轮：优先处理context正确的实体 ==========
    for ent in sorted_entities:
        text_to_find = ent.get('text', '')
        label = ent.get('label', '')
        context_anchor = ent.get('context', '')
        
        if not text_to_find or not label:
            continue
        
        # 检查context是否包含实体text（验证context有效性）
        context_is_valid = context_anchor and text_to_find in context_anchor
        
        # --- 阶段 A: 定位上下文 (Anchor Search) ---
        # 只有context有效时才使用context定位
        if context_is_valid:
            anchor_matches = list(re.finditer(re.escape(context_anchor), original_text))
        else:
            # context无效（空或不包含text），留给第二轮处理
            anchor_matches = []
        
        if not anchor_matches:
            # 第一轮失败：context找不到或无效，加入待处理列表
            failed_entities.append(ent)
            continue
        
        # --- 阶段 B: 在上下文内部定位实体 ---
        # 遍历找到的所有上下文（通常只有1个，除非上下文本身也重复了）
        matched = False
        for match in anchor_matches:
            anchor_start, anchor_end = match.span()
            anchor_text = original_text[anchor_start:anchor_end]
            
            # 在这个小小的上下文片段里找实体，这就非常准了
            # rel_start 是相对于 anchor_start 的偏移量
            rel_start = anchor_text.find(text_to_find)
            if rel_start == -1:
                continue  # 理论上不该发生
            
            # 计算出在整篇摘要中的绝对坐标
            abs_start = anchor_start + rel_start
            abs_end = abs_start + len(text_to_find)
            
            # --- 阶段 C: 冲突检查 (The Guardrail) ---
            # 1. 掩码检查：如果这个位置已经被之前更长的词占了，跳过
            # 这解决了 "Carbon coated" 里的 "Carbon" 被重复标的问题
            if any(char_mask[abs_start:abs_end]):
                continue
            
            # 2. 单词边界检查：防止匹配到单词内部 (如 'Na' 匹配到 'Nature')
            #    但对于化学式等特殊情况放宽限制
            is_word_boundary = True
            
            # 左边界检查
            if abs_start > 0:
                left_char = original_text[abs_start-1]
                first_char = text_to_find[0]
                # 如果左边是字母数字，且实体不是以大写字母/数字开头，则不满足边界
                if left_char.isalnum() and not (first_char.isupper() or first_char.isdigit()):
                    is_word_boundary = False
            
            # 右边界检查
            if abs_end < len(original_text):
                right_char = original_text[abs_end]
                last_char = text_to_find[-1]
                # 如果右边是字母数字，且实体不是以 ) ] 数字 结尾，则不满足边界
                # 化学式如 NaTi2(PO4)(3) 以 ) 结尾，后面可能紧跟 nanocrystals
                if right_char.isalnum() and last_char not in ')]}0123456789':
                    is_word_boundary = False
            
            if not is_word_boundary:
                continue
            
            # --- 阶段 D: 生成结果并占位 ---
            ls_result = {
                "value": {
                    "start": abs_start,
                    "end": abs_end,
                    "text": original_text[abs_start:abs_end],  # 使用原文精确文本
                    "labels": [label]
                },
                "id": generate_ls_id(),
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "llm-annotation"
            }
            results.append(ls_result)
            
            # 标记这些位置已被占用
            for i in range(abs_start, abs_end):
                char_mask[i] = 1
            
            matched = True
            # 只要匹配成功一次，就跳出当前上下文循环
            break
        
        if not matched:
            # 第一轮未匹配成功，加入待处理列表（区别于已找到但被占用的情况）
            failed_entities.append(ent)
    
    # ========== 第二轮：处理context无效的实体，使用全文搜索 + 掩码 ==========
    for ent in failed_entities:
        text_to_find = ent.get('text', '')
        label = ent.get('label', '')
        
        if not text_to_find or not label:
            continue
        
        # 全文搜索所有匹配位置
        all_matches = list(re.finditer(re.escape(text_to_find), original_text))
        
        if not all_matches:
            print(f"    警告: 实体 '{text_to_find}' 在文本中未找到")
            continue
        
        # 尝试每个匹配位置（按原文顺序）
        matched = False
        for match in all_matches:
            abs_start, abs_end = match.span()
            
            # 掩码检查：跳过已被占用的位置
            if any(char_mask[abs_start:abs_end]):
                continue
            
            # 单词边界检查
            is_word_boundary = True
            
            # 左边界检查
            if abs_start > 0:
                left_char = original_text[abs_start-1]
                first_char = text_to_find[0]
                if left_char.isalnum() and not (first_char.isupper() or first_char.isdigit()):
                    is_word_boundary = False
            
            # 右边界检查
            if abs_end < len(original_text):
                right_char = original_text[abs_end]
                last_char = text_to_find[-1]
                if right_char.isalnum() and last_char not in ')]}0123456789':
                    is_word_boundary = False
            
            if not is_word_boundary:
                continue
            
            # 生成结果并占位
            ls_result = {
                "value": {
                    "start": abs_start,
                    "end": abs_end,
                    "text": original_text[abs_start:abs_end],
                    "labels": [label]
                },
                "id": generate_ls_id(),
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "llm-annotation-fallback"  # 标记为fallback
            }
            results.append(ls_result)
            
            # 标记已占用
            for i in range(abs_start, abs_end):
                char_mask[i] = 1
            
            matched = True
            break
        
        if not matched:
            print(f"    警告: 实体 '{text_to_find}' 所有位置均被占用或不满足边界条件")
    
    return results


def convert_llm_annotation_to_labelstudio(llm_result):
    """
    将单个 LLM 标注结果转换为 Label Studio 格式
    
    Args:
        llm_result: LLM 标注结果（包含 abstract, entities 等）
        
    Returns:
        ls_format: Label Studio 格式的标注数据
    """
    abstract = llm_result.get('abstract', '')
    entities = llm_result.get('entities', [])
    title = llm_result.get('title', '')
    
    if entities is None:
        entities = []
    
    # 使用上下文锚点的精准对齐算法
    result = align_predictions(abstract, entities)
    
    # 构建完整的 Label Studio 格式
    ls_format = {
        "annotations": [
            {
                "id": 0,
                "completed_by": "llm",
                "result": result,
                "was_cancelled": False,
                "ground_truth": False,
                "created_at": datetime.now().isoformat(),
                "lead_time": 0,
                "result_count": len(result)
            }
        ],
        "data": {
            "text": abstract,  # 使用 text 字段，与原始训练数据格式一致
            "abstract": abstract,
            "title": title
        },
        "meta": {
            "source": "llm-annotation",
            "round": llm_result.get('sampling_info', {}).get('method', 'unknown')
        }
    }
    
    return ls_format


def convert_round_annotations(round_num):
    """
    转换指定轮次的所有 LLM 标注结果
    
    Args:
        round_num: 轮次号
        
    Returns:
        converted: 转换后的 Label Studio 格式数据列表
    """
    # 加载 LLM 标注结果
    annotation_file = os.path.join(ANNOTATION_OUTPUT_DIR, f"round_{round_num}_annotated.json")
    
    if not os.path.exists(annotation_file):
        print(f"错误: 找不到标注文件 {annotation_file}")
        return None
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotation_data = json.load(f)
    
    results = annotation_data.get('results', [])
    print(f"加载 {len(results)} 个 LLM 标注结果")
    
    # 转换每个标注
    converted = []
    success_count = 0
    total_entities = 0
    aligned_entities = 0
    
    for i, result in enumerate(results):
        if not result.get('success', False) or result.get('entities') is None:
            print(f"  跳过失败样本: {result.get('title', '')[:50]}")
            continue

        # 统计 LLM 原始实体数量（用于计算对齐率）
        total_entities += len(result.get('entities', []))
        
        ls_format = convert_llm_annotation_to_labelstudio(result)

        # 统计成功对齐到文本位置的实体数量
        aligned_entities += len(ls_format['annotations'][0]['result'])
        
        # 添加唯一 ID
        ls_format['id'] = f"llm_r{round_num}_{i}"
        
        converted.append(ls_format)
        success_count += 1
    
    print(f"成功转换 {success_count} 个样本")
    if total_entities > 0:
        alignment_rate = aligned_entities / total_entities * 100
        print(f"实体对齐率: {aligned_entities}/{total_entities} ({alignment_rate:.1f}%)")
    else:
        print("实体对齐率: 无实体可统计")
    
    # 保存转换结果
    output_file = os.path.join(ANNOTATION_OUTPUT_DIR, f"round_{round_num}_labelstudio.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    
    print(f"保存至: {output_file}")
    
    return converted


def merge_with_training_data(round_num, original_train_file, output_file):
    """
    将 LLM 标注数据与原始训练数据合并
    
    Args:
        round_num: 轮次号
        original_train_file: 原始训练数据文件路径
        output_file: 输出文件路径
        
    Returns:
        merged: 合并后的数据
    """
    # 加载 Label Studio 格式的 LLM 标注
    ls_file = os.path.join(ANNOTATION_OUTPUT_DIR, f"round_{round_num}_labelstudio.json")
    
    if not os.path.exists(ls_file):
        print("先转换 LLM 标注结果...")
        convert_round_annotations(round_num)
    
    with open(ls_file, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)
    
    # 加载原始训练数据
    with open(original_train_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print(f"原始训练数据: {len(original_data)} 样本")
    print(f"LLM 标注数据: {len(llm_data)} 样本")
    
    # 合并
    merged = original_data + llm_data
    
    print(f"合并后: {len(merged)} 样本")
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    print(f"保存至: {output_file}")
    
    return merged


if __name__ == '__main__':
    import sys
    
    round_num = 1
    if len(sys.argv) > 1:
        round_num = int(sys.argv[1])
    
    # 转换 LLM 标注
    print("=" * 60)
    print(f"转换 Round {round_num} LLM 标注结果")
    print("=" * 60)
    
    converted = convert_round_annotations(round_num)
    
    if converted:
        # 统计
        total_entities = sum(
            len(item['annotations'][0]['result']) 
            for item in converted
        )
        print(f"\n转换后实体总数: {total_entities}")
