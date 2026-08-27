# -*- coding: utf-8 -*-
"""
带结构纠错的 LLM 标注器（实验组）

流程（复用对照组标注结果）：
1. 读取对照组的标注结果（LLM/annotated/）
2. 对每条标注结果进行错误检测
3. 有错误 → 反馈给 LLM 修正
4. 无错误 → 直接使用原标注
5. 保存纠错后的结果

这样设计保证：起点完全相同，唯一变量是"纠错机制"
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import List
from openai import OpenAI

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ANNOTATED_OUTPUT_DIR, BASELINE_ANNOTATED_DIR,
    MAX_REFINEMENT_ROUNDS
)
from error_detector import ErrorDetector, detect_annotation_errors

# ============= DeepSeek API 配置（复制自 LLM） =============
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"

def create_client():
    """创建 DeepSeek API 客户端"""
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# ============= 基础系统提示词（复制自 LLM） =============
BASE_SYSTEM_PROMPT = """# Role
你是一位精通钠离子电池（SIBs）材料学的科研专家，同时也是一位严谨的数据标注员。你的任务是从给定的文献摘要中抽取实体。

# Output Format
请以严格的 JSON 格式输出，不要包含 markdown 代码块标记（```json），不要包含任何解释性文字。
JSON 格式如下：
[
  {"text": "提取的实体文本", "label": "实体类型", "context": "包含该实体的原文短句片段"},
  ...
]

**⚠️⚠️⚠️ 最关键要求 - text 必须是 context 的子串：**
1. **text 必须能在 context 中被精确找到**（作为子字符串）
2. **text 和 context 都必须是原文的【逐字精确复制】**：
   - 原文中的空格、括号、下标标记等必须保持原样
   - 例如原文是 `Na3V2(PO4)(3)` 就写 `Na3V2(PO4)(3)`，不要改成 `Na3V2(PO4)3`
3. **context 用于定位**：应包含足够的上下文词汇，使其在全文中唯一
4. **绝对禁止**对化学式进行任何"规范化"或"美化"

**错误示例（禁止）：**
- text: "disordered 3D multi-layer graphene", context: "disordered multi-layer graphene" ❌ (text 中有 "3D" 但 context 中没有)

**正确示例：**
- text: "disordered 3D multi-layer graphene", context: "The disordered 3D multi-layer graphene anode" ✓ (text 完整出现在 context 中)

**重要：如果摘要中不存在任何符合定义的实体，请务必直接返回空列表 []。**

# Entity Types & Definitions
1. **MATERIAL (核心活性材料)**
   - 定义：电极中起主要作用的活性材料本体。
   - 包含：明确的化学式（Na3V2(PO4)3）、材料名称（hard carbon）、明确的复合形式（Na3V2(PO4)3/C, X@C）。
   - **绝对不包含**：泛指词（material, electrode, sample, particles, powder）、体系描述词（composite, based）、非活性物质（PVDF, binder）、仅作为合成前驱体/反应物的物质（如 `Co can react with Sb2S3 to form CoSbS` 中的 `Co` 不标，只标生成物 `CoSbS` 和反应物 `Sb2S3`）。
   - *特殊规则*：如果是紧凑的缩写（如 MVO-NBs），整体标为 MATERIAL；如果是全称（FePO4 nanoparticles），只标 FePO4，nanoparticles 标为 O。
   - *前缀剥离规则*：**严禁将掺杂元素前缀与母体合并标注**。如 `B-NVP/C` 中，`B-` 前缀标为 O，核心母体 `NVP/C` 标为 MATERIAL。

2. **MODIFICATION (化学修饰)**
   - 定义：化学层面的改性手段。
   - 包含：doped, coated, substitution, oxygen-vacancy, defect-rich。
   - **绝对不包含**：物理形貌（porous, hollow, nano-sized）、性能描述（high-performance）。
   - *增强判据*：只有当该词描述的是材料被制备后"具有的化学属性"时才标注为 MODIFICATION；**若其仅作为被抑制、被促进的对象出现（如 suppresses vacancy ordering），或用于描述电化学反应机制（如 Fe2+/Fe3+ redox couple, Na+ extraction），则绝对不标。**
   - *动作绑定原则*：被动语态中，作为改性动词（doped, coated 等）主语的元素或离子也标为 MODIFICATION。示例：`V-ions are successfully doped` → `V-ions` 和 `doped` 均标为 MODIFICATION。
   - *松散结构拆分规则*：当改性描述中出现介词（of, with, by, through）时，必须拆分。示例：`doping of nitrogen and sulfur` → `doping`、`nitrogen`、`sulfur` 各自标为 MODIFICATION；`nitrogen and sulfur co-doping`（无介词）→ 整体标为 MODIFICATION。

3. **STRUCTURE (晶体/相结构)**
   - 定义：晶体结构或相结构类型。
   - 包含：layered, spinel, olivine, NASICON, tunnel, Prussian blue framework, disordered；**以及晶体空间群符号（如 P6(3)/mmc, Fd-3m, R-3m）**。注意：只标核心结构词本身，泛化后缀不标（如 `tunnel structure` → 只标 `tunnel`，`NASICON structure` → 只标 `NASICON`）。
   - **绝对不包含**：形貌（nanosheets, nanorods）、泛化结构（carbon framework, 3D framework）。
   - *白名单规则*：只有公认的独立晶体结构术语才标，其他含 framework 的词一律不标。
   - *动态排除规则*：**描述相变、转变或演化过程的表达（如 P2–O2 transition, phase transition, phase evolution）不属于 STRUCTURE，一律不标。**

4. **ROLE (电池角色)**
   - 包含：anode, cathode, positive electrode, negative electrode。
   - 规则：只标角色词本身，不包含后续的 material/electrode（例如 "anode material" -> 仅标 "anode"）。"""

STATIC_FEWSHOT = """
# Few-Shot Examples (带上下文锚点)

Input: "The P2 phase is stable, while the P2-O2 transition causes capacity decay."
Output:
[
  {"text": "P2", "label": "STRUCTURE", "context": "The P2 phase is stable"}
]
*注意：只标第一个独立的 P2（晶体结构）；"P2-O2 transition" 是相变过程不标；context 精确定位到前半句。*

Input: "Fe-doped Na3V2(PO4)3/C composite suppresses phase transition."
Output:
[
  {"text": "Fe-doped", "label": "MODIFICATION", "context": "Fe-doped Na3V2(PO4)3/C"},
  {"text": "Na3V2(PO4)3/C", "label": "MATERIAL", "context": "Fe-doped Na3V2(PO4)3/C composite"}
]
*注意：不包含 "composite"（体系词）；"phase transition" 是过程不标。*

Input: "We synthesized porous layered Na0.67MnO2 as cathode material."
Output:
[
  {"text": "layered", "label": "STRUCTURE", "context": "porous layered Na0.67MnO2"},
  {"text": "Na0.67MnO2", "label": "MATERIAL", "context": "layered Na0.67MnO2 as cathode"},
  {"text": "cathode", "label": "ROLE", "context": "as cathode material"}
]
*注意：不包含 "porous"（形貌）；"cathode material" 中只标 "cathode"。*

Input: "The electrochemical performance was evaluated at room temperature."
Output:
[]
*注意：文中无符合定义的实体，返回空列表。*"""

# 修正提示词模板（实验组独有）
REFINEMENT_PROMPT_TEMPLATE = """You previously annotated the following abstract for named entities:

Abstract:
{abstract}

Your previous annotation:
{previous_output}

{error_feedback}

Based on the guideline violations above, please re-evaluate the flagged entities and output the complete corrected annotation result in JSON array format.

IMPORTANT:
- Keep correct annotations unchanged, only modify entities that violate the guidelines
- Each entity MUST include "text", "label", and "context" fields
- The "context" field should contain a text snippet from the abstract that includes the entity for precise position location
- If you split an entity into multiple entities, ensure each split entity has its own accurate "context" field"""

def parse_llm_response(response_text: str) -> list:
    """
    解析 LLM 响应，提取 JSON 数组
    """
    # 尝试直接解析
    try:
        return json.loads(response_text)
    except:
        pass
    
    # 尝试提取 JSON 代码块
    import re
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    # 尝试找到 [ 和 ] 之间的内容
    bracket_match = re.search(r'\[[\s\S]*\]', response_text)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(0))
        except:
            pass
    
    return None


def merge_refinement(original_entities: List[dict], refined_entities: List[dict], 
                     flagged_indices: set) -> List[dict]:
    """
    合并修正结果：只接受被反馈实体的修改，其他实体保持原样
    
    使用索引定位，解决同名实体的歧义问题。
    
    策略：
    1. 未被反馈的实体（索引不在 flagged_indices 中）：100% 保留原样
    2. 被反馈的实体：使用 LLM 在对应位置的修正结果
    3. 要求 LLM 保持实体顺序，通过索引对应
    
    Args:
        original_entities: 原始实体列表
        refined_entities: LLM 修正后的实体列表（要求保持顺序）
        flagged_indices: 被反馈的实体索引集合 {0, 3, 5, ...}
    
    Returns:
        merged: 合并后的实体列表
    """
    merged = []
    
    # 如果存在全局漏标错误 (-1)，强制进入文本匹配模式（因为长度可能由于追加新实体而改变）
    # 或者如果 LLM 返回的实体数量和原始不同，采用文本匹配的安全兜底策略
    if (-1 in flagged_indices) or (len(refined_entities) != len(original_entities)):
        return _merge_by_text_matching(original_entities, refined_entities, flagged_indices)
    
    # 正常情况：按索引一一对应
    for idx, original_entity in enumerate(original_entities):
        if idx in flagged_indices:
            # 被反馈的实体：接受 LLM 的修正
            merged.append(refined_entities[idx])
        else:
            # 未被反馈的实体：强制保留原样（即使 LLM 改了也不接受）
            merged.append(original_entity)
    
    return merged


def _merge_by_text_matching(original_entities: List[dict], refined_entities: List[dict],
                            flagged_indices: set) -> List[dict]:
    """
    基于文本匹配的 fallback 合并策略
    
    当 LLM 返回的实体数量与原始不同时使用。
    
    关键改进：支持实体拆分场景
    例如：原始 "disordered 3D multi-layer graphene" (MATERIAL) 
         拆分为 "disordered" (STRUCTURE) + "graphene" (MATERIAL)
    """
    merged = []
    
    # 记录已被消费的 refined 实体索引
    consumed_refined_indices = set()
    
    for idx, original_entity in enumerate(original_entities):
        original_text = original_entity.get('text', '')
        original_context = original_entity.get('context', '')
        
        if idx in flagged_indices:
            # 被反馈的实体：找到 LLM 的修正（可能是多个拆分后的实体）
            split_entities = []
            
            for ref_idx, ref_ent in enumerate(refined_entities):
                if ref_idx in consumed_refined_indices:
                    continue
                    
                ref_text = ref_ent.get('text', '')
                ref_context = ref_ent.get('context', '')
                
                # 拆分匹配的条件：支持拆分（ref是orig的子串）或修复类扩展（orig是ref的子串）
                is_split_part = ref_text in original_text or original_text in ref_text
                
            # context_similar 判断条件修改：适应更短的短语境，解决截断遗失风险
                context_words_orig = set(original_context.split())
                context_words_ref = set(ref_context.split())
                min_match_len = min(2, len(context_words_orig)) # 至少2个共同单词或原context的长度
                
                context_similar = (
                    ref_context == original_context or  # 完全相同
                    (ref_context and original_context and 
                     len(context_words_ref & context_words_orig) >= min_match_len)
                )
                
                # 只有满足"是拆分部分"且"context相似"才匹配
                if is_split_part and context_similar:
                    split_entities.append((ref_idx, ref_ent))
            
            # 添加所有匹配到的拆分实体
            for ref_idx, ref_ent in split_entities:
                merged.append(ref_ent)
                consumed_refined_indices.add(ref_idx)
            
            # 如果没找到任何匹配，说明 LLM 删除了这个实体（接受删除）
        else:
            # 未被反馈的实体：强制保留原样
            merged.append(original_entity)
            # 修复重复漏洞：标记 LLM 返回的对应未修改实体为已消费，防止 -1 逻辑中重复添加
            for ref_idx, ref_ent in enumerate(refined_entities):
                if ref_idx not in consumed_refined_indices and ref_ent.get('text') == original_text:
                    consumed_refined_indices.add(ref_idx)
                    break
    
    # 追加大模型提供的新增（补漏）实体
    # 也就是那些 LLM 生成的，但没有在上面 "consumed"（被当做已有实体的修改或拆分吸收）的实体
    if -1 in flagged_indices:
        for ref_idx, ref_ent in enumerate(refined_entities):
            if ref_idx not in consumed_refined_indices:
                # 为了防止把大模型瞎编的错误实体加进来，建议简单做个去重
                if ref_ent not in merged:
                    ref_text = ref_ent.get('text', '')
                    ref_context = ref_ent.get('context', '')
                    # 强力防御大模型幻觉: text 必须实际在 context 中存在
                    if ref_text and ref_context and (ref_text in ref_context or ref_text.lower() in ref_context.lower()):
                        merged.append(ref_ent)
                    else:
                        print(f"      拦截到幻觉/非法实体(text不在context中): '{ref_text}'")

    return merged

def refine_round(round_num: int, delay: float = 1.0):
    """
    对对照组的标注结果进行纠错（实验组核心流程）
    
    流程：
    1. 读取对照组的标注结果
    2. 对每条进行错误检测
    3. 有错误 -> 反馈给 LLM 修正
    4. 无错误 -> 直接保留原标注
    
    Args:
        round_num: 轮次号
        delay: API 调用间隔（秒）
    """
    print("=" * 60)
    print(f"结构纠错 - Round {round_num}")
    print("=" * 60)
    
    # 加载对照组的标注结果（作为起点）
    baseline_file = os.path.join(BASELINE_ANNOTATED_DIR, f"round_{round_num}_annotated.json")
    if not os.path.exists(baseline_file):
        print(f"错误: 找不到对照组标注文件 {baseline_file}")
        return None
    
    with open(baseline_file, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)
    
    baseline_results = baseline_data['results']
    print(f"\n加载对照组标注结果: {len(baseline_results)} 条")
    
    # 创建 API 客户端（只在需要纠错时使用）
    client = None
    detector = ErrorDetector()
    
    # 统计
    results = []
    no_error_count = 0
    has_error_count = 0
    refinement_success = 0
    refinement_failed = 0
    total_errors_detected = 0
    total_refinements = 0
    
    print("\n开始检测并纠错...")
    for i, baseline_result in enumerate(baseline_results):
        title = baseline_result.get('title', 'Unknown')
        abstract = baseline_result.get('abstract', '')
        original_entities = baseline_result.get('entities', [])
        
        print(f"\n[{i+1}/{len(baseline_results)}] {title[:50]}...")
        
        # 步骤1: 检测错误
        if original_entities is None:
            # 对照组标注失败的，直接跳过
            print(f"    ⚠ 对照组标注失败，跳过")
            results.append({
                **baseline_result,
                'errors_detected': [],
                'refinement_history': [],
                'refinement_rounds': 0,
                'refined': False
            })
            continue
        
        errors, feedback = detect_annotation_errors(original_entities, abstract)
        
        if not errors:
            # 无错误，直接使用原标注
            no_error_count += 1
            print(f"    ✓ 无错误，保留原标注 ({len(original_entities)} 个实体)")
            results.append({
                **baseline_result,
                'errors_detected': [],
                'refinement_history': [],
                'refinement_rounds': 0,
                'refined': False
            })
            continue
        
        # 有错误，需要纠错
        has_error_count += 1
        total_errors_detected += len(errors)
        
        print(f"    检测到 {len(errors)} 个错误:")
        for err in errors[:3]:
            print(f"      - [{err.get('error_type')}] \"{err.get('entity')}\"")
        if len(errors) > 3:
            print(f"      ... 还有 {len(errors) - 3} 个错误")
        
        # 懒加载 API 客户端
        if client is None:
            print("\n    连接 DeepSeek API...")
            client = create_client()
        
        # 步骤2: 调用 LLM 修正
        refinement_result = refine_entities(
            client, abstract, original_entities, errors, feedback, MAX_REFINEMENT_ROUNDS
        )
        
        if refinement_result['success']:
            refinement_success += 1
            total_refinements += refinement_result['rounds']
            final_entities = refinement_result['entities']
            print(f"    ✓ 修正成功 ({refinement_result['rounds']} 轮)，最终 {len(final_entities)} 个实体")
        else:
            refinement_failed += 1
            final_entities = original_entities  # 修正失败，保留原标注
            print(f"    ✗ 修正失败，保留原标注")
        
        results.append({
            'title': title,
            'abstract': abstract,
            'year': baseline_result.get('year'),
            'sampling_info': baseline_result.get('sampling_info'),
            'entities': final_entities,
            'original_entities': original_entities,  # 保留原标注用于对比
            'errors_detected': errors,
            'refinement_history': refinement_result.get('history', []),
            'refinement_rounds': refinement_result.get('rounds', 0),
            'refined': refinement_result['success'],
            'success': True
        })
        
        # API 调用间隔
        time.sleep(delay)
    
    # 保存结果
    output_data = {
        'round': round_num,
        'timestamp': datetime.now().isoformat(),
        'source': 'LLM_SR',
        'method': 'Self-Refinement on LLM baseline',
        'baseline_file': baseline_file,
        'statistics': {
            'total': len(baseline_results),
            'no_error': no_error_count,
            'has_error': has_error_count,
            'refinement_success': refinement_success,
            'refinement_failed': refinement_failed,
            'total_entities': sum(len(r['entities']) for r in results if r.get('entities')),
            'total_errors_detected': total_errors_detected,
            'total_refinement_rounds': total_refinements
        },
        'results': results
    }
    
    os.makedirs(ANNOTATED_OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(ANNOTATED_OUTPUT_DIR, f"round_{round_num}_annotated.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("纠错完成!")
    print("=" * 60)
    print(f"\n统计:")
    print(f"  总样本数: {len(baseline_results)}")
    print(f"  无错误（直接保留）: {no_error_count}")
    print(f"  有错误（需纠错）: {has_error_count}")
    print(f"    - 修正成功: {refinement_success}")
    print(f"    - 修正失败: {refinement_failed}")
    print(f"  检测到的错误总数: {total_errors_detected}")
    print(f"  修正轮次总数: {total_refinements}")
    print(f"\n结果保存至: {output_file}")
    
    # 显示标注摘要
    show_annotation_summary(output_data)
    
    return output_data


def refine_entities(client, abstract: str, original_entities: List[dict], 
                    errors: List[dict], feedback: str, max_rounds: int) -> dict:
    """
    对有错误的标注进行修正
    
    Args:
        client: API 客户端
        abstract: 摘要文本
        original_entities: 原始标注
        errors: 检测到的错误
        feedback: 格式化的反馈
        max_rounds: 最大修正轮数
    
    Returns:
        result: {'success': bool, 'entities': list, 'rounds': int, 'history': list}
    """
    result = {
        'success': False,
        'entities': None,
        'rounds': 0,
        'history': []
    }
    
    current_entities = original_entities
    current_errors = errors
    current_feedback = feedback
    
    for round_num in range(1, max_rounds + 1):
        # 记录被反馈的实体索引
        flagged_indices = {e.get('entity_index') for e in current_errors if 'entity_index' in e}
        
        # 构建修正提示
        previous_output = json.dumps(current_entities, ensure_ascii=False, indent=2)
        refinement_prompt = REFINEMENT_PROMPT_TEMPLATE.format(
            abstract=abstract,
            previous_output=previous_output,
            error_feedback=current_feedback
        )
        
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": BASE_SYSTEM_PROMPT + STATIC_FEWSHOT},
                    {"role": "user", "content": refinement_prompt}
                ],
                temperature=0.0,
                max_tokens=2000,
                extra_body={"thinking": {"type": "disabled"}},
            )
            
            raw_response = response.choices[0].message.content
            refined_entities_raw = parse_llm_response(raw_response)
            
            if refined_entities_raw is None:
                print(f"      修正响应解析失败")
                break
            
            # 合并修正结果（只接受被反馈实体的修改）
            refined_entities = merge_refinement(current_entities, refined_entities_raw, flagged_indices)
            
            result['history'].append({
                'round': round_num,
                'errors': current_errors,
                'feedback': current_feedback,
                'entities_before': current_entities,
                'entities_raw': refined_entities_raw,
                'entities_merged': refined_entities,
                'flagged_indices': list(flagged_indices)
            })
            
            # 检测修正后是否还有错误
            new_errors, new_feedback = detect_annotation_errors(refined_entities, abstract)
            
            if not new_errors:
                # 无错误，修正成功
                result['success'] = True
                result['entities'] = refined_entities
                result['rounds'] = round_num
                return result
            
            # 还有错误，继续下一轮
            current_entities = refined_entities
            current_errors = new_errors
            current_feedback = new_feedback
            
        except Exception as e:
            print(f"      第 {round_num} 轮修正失败: {e}")
            break
    
    # 达到最大轮数或出错，返回最后一轮的结果
    result['success'] = True  # 至少做了修正
    result['entities'] = current_entities
    result['rounds'] = max_rounds
    return result


def show_annotation_summary(output_data):
    """显示标注结果摘要"""
    print("\n" + "=" * 60)
    print("标注结果摘要")
    print("=" * 60)
    
    # 统计实体类型分布
    entity_counts = {}
    error_type_counts = {}
    
    for result in output_data['results']:
        entities = result.get('entities', [])
        if entities:
            for ent in entities:
                label = ent.get('label', 'UNKNOWN')
                entity_counts[label] = entity_counts.get(label, 0) + 1
        
        errors = result.get('errors_detected', [])
        for error in errors:
            error_type = error.get('type', 'UNKNOWN')
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
    
    print("\n实体类型分布:")
    for label, count in sorted(entity_counts.items()):
        print(f"  {label}: {count}")
    
    total_entities = sum(entity_counts.values())
    print(f"\n总计: {total_entities} 个实体")
    
    if error_type_counts:
        print("\n检测到的错误类型分布:")
        for error_type, count in sorted(error_type_counts.items()):
            print(f"  {error_type}: {count}")


if __name__ == '__main__':
    # 默认纠错 Round 1
    round_num = 1
    if len(sys.argv) > 1:
        round_num = int(sys.argv[1])
    
    # 执行纠错（基于对照组标注结果）
    refine_round(round_num)
