# -*- coding: utf-8 -*-
"""
关键词分层采样模块


"""

import json
import os
import re
from datetime import datetime

from config import (
    POOL_FILE, SELECTED_SAMPLES_FILE, SAMPLE_OUTPUT_DIR,
    STRUCTURE_KEYWORDS, MODIFICATION_KEYWORDS,
    STRUCTURE_SAMPLE_NUM, MODIFICATION_SAMPLE_NUM
)


def load_pool():
    """加载样本池"""
    with open(POOL_FILE, 'r', encoding='utf-8') as f:
        pool = json.load(f)
    return pool


def load_selected_samples():
    """加载已选样本ID列表"""
    if os.path.exists(SELECTED_SAMPLES_FILE):
        with open(SELECTED_SAMPLES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"selected_ids": [], "history": []}


def save_selected_samples(selected_data):
    """保存已选样本ID列表"""
    with open(SELECTED_SAMPLES_FILE, 'w', encoding='utf-8') as f:
        json.dump(selected_data, f, ensure_ascii=False, indent=2)


def count_keyword_matches(text, keywords):
    """
    统计文本中关键词的匹配数量（同一关键词多次出现只计1次）

    Returns:
        match_count: 匹配的关键词数量
        matched_keywords: 匹配到的关键词列表
    """
    text_lower = text.lower()
    match_count = 0
    matched_keywords = []

    for keyword in keywords:
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        if re.search(pattern, text_lower):
            match_count += 1
            matched_keywords.append(keyword)

    return match_count, matched_keywords


def keyword_sampling(pool, keywords, num_samples, excluded_ids):
    """
    基于关键词的采样：选取关键词匹配数最多的前 num_samples 篇

    Args:
        pool: 样本池
        keywords: 关键词列表
        num_samples: 采样数量
        excluded_ids: 需要排除的样本ID（已选或本轮另一类已选）

    Returns:
        selected: 选中的样本列表
        selected_ids: 选中的样本ID列表
    """
    candidates = []

    for idx, sample in enumerate(pool):
        sample_id = sample.get('id', idx)
        if sample_id in excluded_ids:
            continue

        text = sample.get('abstract', '')
        if not text:
            continue

        match_count, matched_keywords = count_keyword_matches(text, keywords)

        if match_count > 0:
            candidates.append({
                'sample': sample,
                'id': sample_id,
                'match_count': match_count,
                'matched_keywords': matched_keywords
            })

    candidates.sort(key=lambda x: x['match_count'], reverse=True)

    selected = []
    selected_ids = []

    for cand in candidates[:num_samples]:
        selected.append({
            **cand['sample'],
            '_sampling_info': {
                'method': 'keyword',
                'match_count': cand['match_count'],
                'matched_keywords': cand['matched_keywords']
            }
        })
        selected_ids.append(cand['id'])

    return selected, selected_ids


def keyword_sample(round_num=None):
    """
    执行一轮关键词分层采样

    Args:
        round_num: 轮次号（如果为None则自动递增）

    Returns:
        round_output: 本轮采样结果
    """
    print("=" * 60)
    print("关键词分层采样")
    print("=" * 60)

    print("\n1. 加载样本池...")
    pool = load_pool()
    print(f"   样本池大小: {len(pool)}")

    selected_data = load_selected_samples()
    excluded_ids = set(selected_data['selected_ids'])
    print(f"   历史已选样本数: {len(excluded_ids)}")

    available_count = len(pool) - len(excluded_ids)
    print(f"   当前可用样本数: {available_count}")

    if available_count < STRUCTURE_SAMPLE_NUM + MODIFICATION_SAMPLE_NUM:
        print(f"   警告: 可用样本不足，将采样所有剩余样本")

    if round_num is None:
        round_num = len(selected_data['history']) + 1
    print(f"\n   当前轮次: Round {round_num}")

    # ========== Step 1: STRUCTURE 关键词采样 ==========
    print(f"\n2. STRUCTURE 关键词采样 (目标: {STRUCTURE_SAMPLE_NUM}篇)...")
    structure_selected, structure_ids = keyword_sampling(
        pool, STRUCTURE_KEYWORDS, STRUCTURE_SAMPLE_NUM, excluded_ids
    )
    print(f"   实际采样: {len(structure_selected)}篇")

    excluded_ids.update(structure_ids)

    # ========== Step 2: MODIFICATION 关键词采样 ==========
    print(f"\n3. MODIFICATION 关键词采样 (目标: {MODIFICATION_SAMPLE_NUM}篇)...")
    modification_selected, modification_ids = keyword_sampling(
        pool, MODIFICATION_KEYWORDS, MODIFICATION_SAMPLE_NUM, excluded_ids
    )
    print(f"   实际采样: {len(modification_selected)}篇")

    # ========== 汇总结果 ==========
    all_selected = structure_selected + modification_selected
    all_ids = structure_ids + modification_ids

    print(f"\n4. 采样汇总:")
    print(f"   STRUCTURE:    {len(structure_selected)}篇")
    print(f"   MODIFICATION: {len(modification_selected)}篇")
    print(f"   总计:         {len(all_selected)}篇")

    # ========== 保存结果 ==========
    print("\n5. 保存结果...")
    os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)

    round_output = {
        'round': round_num,
        'timestamp': datetime.now().isoformat(),
        'statistics': {
            'structure': len(structure_selected),
            'modification': len(modification_selected),
            'total': len(all_selected)
        },
        'samples': all_selected
    }

    output_file = os.path.join(SAMPLE_OUTPUT_DIR, f"round_{round_num}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(round_output, f, ensure_ascii=False, indent=2)
    print(f"   本轮采样保存至: {output_file}")

    selected_data['selected_ids'].extend(all_ids)
    selected_data['history'].append({
        'round': round_num,
        'timestamp': datetime.now().isoformat(),
        'count': len(all_selected),
        'ids': all_ids
    })
    save_selected_samples(selected_data)
    print(f"   已选样本记录更新: {SELECTED_SAMPLES_FILE}")

    print(f"\n   累计已选样本数: {len(selected_data['selected_ids'])}")
    print(f"   剩余可用样本数: {len(pool) - len(selected_data['selected_ids'])}")

    print("\n" + "=" * 60)
    print("采样完成!")
    print("=" * 60)

    return round_output


def show_sampling_details(round_output):
    """显示采样详情"""
    print("\n" + "=" * 60)
    print("采样详情")
    print("=" * 60)

    for i, sample in enumerate(round_output['samples']):
        info = sample.get('_sampling_info', {})
        text = sample.get('abstract', '')
        text_preview = text[:100] + "..." if len(text) > 100 else text

        print(f"\n[{i+1}] 匹配数: {info.get('match_count')}")
        print(f"    关键词: {info.get('matched_keywords')}")
        print(f"    摘要: {text_preview}")


if __name__ == '__main__':
    import sys

    round_num = None
    if len(sys.argv) > 1:
        round_num = int(sys.argv[1])

    result = keyword_sample(round_num)

    show_details = input("\n是否显示采样详情? (y/n): ").strip().lower()
    if show_details == 'y':
        show_sampling_details(result)
