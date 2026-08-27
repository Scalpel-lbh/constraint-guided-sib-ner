# -*- coding: utf-8 -*-
"""
LLM 自动标注模块 - 使用 DeepSeek 进行 NER 标注
支持 KNN 动态检索 few-shot 示例
"""

import json
import os
import time
from datetime import datetime
from openai import OpenAI

from config import SAMPLE_OUTPUT_DIR

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"  
#DEEPSEEK_MODEL = "deepseek-reasoner"  # R1 推理模型（更准确但更慢）

# KNN 检索配置
USE_KNN_FEWSHOT = True  # 启用 KNN 动态检索 few-shot
KNN_K = 3  # 检索的示例数量

# 标注输出目录
ANNOTATION_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "annotated")

# 基础系统提示词（不含 few-shot 示例）
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

# 静态 few-shot 示例（当不使用 KNN 时使用）
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

# KNN 检索器（延迟初始化）
_knn_retriever = None

def get_knn_retriever():
    """获取或初始化 KNN 检索器"""
    global _knn_retriever
    if _knn_retriever is None and USE_KNN_FEWSHOT:
        from knn_retriever import KNNRetriever
        _knn_retriever = KNNRetriever(k=KNN_K)
    return _knn_retriever

def build_system_prompt(abstract: str = None) -> str:
    """
    构建系统提示词
    
    Args:
        abstract: 待标注的摘要（用于 KNN 检索）
        
    Returns:
        完整的系统提示词（静态示例 + KNN 动态示例）
    """
    # 基础 prompt + 静态 few-shot（始终包含）
    prompt = BASE_SYSTEM_PROMPT + STATIC_FEWSHOT
    
    # 如果启用 KNN，追加动态检索的示例
    if USE_KNN_FEWSHOT and abstract:
        retriever = get_knn_retriever()
        if retriever:
            # 从 25 篇人工标注样本中检索相似示例
            examples = retriever.retrieve(abstract)
            
            # 构建动态 few-shot（追加到静态示例之后）
            knn_fewshot_str = "\n\n# Additional Examples (从人工标注训练集中检索)\n以下是与待标注文本最相似的人工标注示例，请重点参考这些标注风格：\n"
            
            for i, ex in enumerate(examples, 1):
                entities_json = json.dumps(ex['entities'], ensure_ascii=False, indent=2)
                # 截断过长的文本
                text_preview = ex['text'][:800] + "..." if len(ex['text']) > 800 else ex['text']
                knn_fewshot_str += f"""
**检索示例 {i}** (相似度: {ex['similarity']:.3f})
Input: "{text_preview}"
Output:
{entities_json}
"""
            
            prompt += knn_fewshot_str
    
    return prompt

def create_client():
    """创建 DeepSeek API 客户端"""
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )
    return client


def annotate_abstract(client, abstract):
    """
    使用 DeepSeek 标注单个摘要
    
    Args:
        client: OpenAI 客户端
        abstract: 摘要文本
        
    Returns:
        entities: 提取的实体列表
        raw_response: 原始响应文本
    """
    # 构建系统提示词（可能包含 KNN 检索的动态 few-shot）
    system_prompt = build_system_prompt(abstract)
    user_prompt = f"Input: \"{abstract}\"\nOutput:"
    
    try:
        # 根据模型类型调整参数
        if DEEPSEEK_MODEL == "deepseek-reasoner":
            # R1 模型不支持 system prompt 和 temperature
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "user", "content": system_prompt + "\n\n" + user_prompt}
                ],
                max_tokens=8000  # R1 需要更多 token 因为包含推理过程
            )
        else:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # 降低随机性，提高一致性
                max_tokens=2000,
                extra_body={"thinking": {"type": "disabled"}},
            )
        
        # R1 模型返回的 content 就是最终结果（reasoning_content 是推理过程）
        raw_response = response.choices[0].message.content
        if raw_response is None:
            raw_response = ""
        raw_response = raw_response.strip()
        
        # 解析 JSON
        try:
            # 清理可能的 markdown 代码块标记
            clean_response = raw_response
            if clean_response.startswith("```"):
                clean_response = clean_response.split("\n", 1)[1]
            if clean_response.endswith("```"):
                clean_response = clean_response.rsplit("```", 1)[0]
            clean_response = clean_response.strip()
            
            # 尝试提取 JSON 数组（如果响应中包含其他文本）
            if not clean_response.startswith("["):
                import re
                # 查找 JSON 数组
                json_match = re.search(r'\[[\s\S]*\]', clean_response)
                if json_match:
                    clean_response = json_match.group(0)
            
            entities = json.loads(clean_response)
            return entities, raw_response
        except json.JSONDecodeError as e:
            print(f"    JSON 解析错误: {e}")
            print(f"    原始响应: {raw_response[:500] if raw_response else 'Empty'}...")
            return None, raw_response
            
    except Exception as e:
        print(f"    API 调用错误: {e}")
        return None, str(e)


def annotate_round(round_num, retry_failed=True, delay=1.0):
    """
    标注指定轮次的所有样本
    
    Args:
        round_num: 轮次号
        retry_failed: 是否重试失败的样本
        delay: API 调用间隔（秒）
        
    Returns:
        results: 标注结果
    """
    print("=" * 60)
    print(f"LLM 自动标注 - Round {round_num}")
    print("=" * 60)
    
    # 加载采样结果
    sample_file = os.path.join(SAMPLE_OUTPUT_DIR, f"round_{round_num}.json")
    if not os.path.exists(sample_file):
        print(f"错误: 找不到采样文件 {sample_file}")
        return None
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
    
    samples = sample_data['samples']
    print(f"\n加载 {len(samples)} 个样本")
    
    # 创建 API 客户端
    print("\n连接 DeepSeek API...")
    client = create_client()
    
    # 标注每个样本
    results = []
    success_count = 0
    failed_count = 0
    
    print("\n开始标注...")
    for i, sample in enumerate(samples):
        abstract = sample.get('abstract', '')
        title = sample.get('title', 'Unknown')
        
        print(f"\n[{i+1}/{len(samples)}] {title[:50]}...")
        
        entities, raw_response = annotate_abstract(client, abstract)
        
        if entities is not None:
            success_count += 1
            print(f"    ✓ 提取到 {len(entities)} 个实体")
            
            # 显示提取的实体
            for ent in entities[:5]:  # 最多显示5个
                print(f"      - {ent.get('text', '')} [{ent.get('label', '')}]")
            if len(entities) > 5:
                print(f"      ... 还有 {len(entities) - 5} 个实体")
        else:
            failed_count += 1
            print(f"    ✗ 标注失败")
        
        results.append({
            'title': title,
            'abstract': abstract,
            'year': sample.get('year'),
            'sampling_info': sample.get('_sampling_info'),
            'entities': entities,
            'raw_response': raw_response,
            'success': entities is not None
        })
        
        # API 调用间隔
        if i < len(samples) - 1:
            time.sleep(delay)
    
    # 重试失败的样本
    if retry_failed and failed_count > 0:
        print(f"\n重试 {failed_count} 个失败的样本...")
        for i, result in enumerate(results):
            if not result['success']:
                print(f"\n  重试: {result['title'][:50]}...")
                time.sleep(delay * 2)  # 重试时等待更长时间
                
                entities, raw_response = annotate_abstract(client, result['abstract'])
                
                if entities is not None:
                    result['entities'] = entities
                    result['raw_response'] = raw_response
                    result['success'] = True
                    success_count += 1
                    failed_count -= 1
                    print(f"    ✓ 重试成功，提取到 {len(entities)} 个实体")
                else:
                    print(f"    ✗ 重试仍然失败")
    
    # 保存结果
    os.makedirs(ANNOTATION_OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(ANNOTATION_OUTPUT_DIR, f"round_{round_num}_annotated.json")
    
    output_data = {
        'round': round_num,
        'timestamp': datetime.now().isoformat(),
        'statistics': {
            'total': len(samples),
            'success': success_count,
            'failed': failed_count,
            'total_entities': sum(len(r['entities']) for r in results if r['entities'])
        },
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("标注完成!")
    print("=" * 60)
    print(f"\n统计:")
    print(f"  总样本数: {len(samples)}")
    print(f"  成功: {success_count}")
    print(f"  失败: {failed_count}")
    print(f"  提取实体总数: {output_data['statistics']['total_entities']}")
    print(f"\n结果保存至: {output_file}")
    
    return output_data


def show_annotation_summary(output_data):
    """显示标注摘要"""
    print("\n" + "=" * 60)
    print("标注结果摘要")
    print("=" * 60)
    
    # 统计各类型实体数量
    entity_counts = {'MATERIAL': 0, 'STRUCTURE': 0, 'MODIFICATION': 0, 'ROLE': 0}
    
    for result in output_data['results']:
        if result['entities']:
            for ent in result['entities']:
                label = ent.get('label', '')
                if label in entity_counts:
                    entity_counts[label] += 1
    
    print("\n实体类型分布:")
    for label, count in entity_counts.items():
        print(f"  {label}: {count}")
    
    print(f"\n总计: {sum(entity_counts.values())} 个实体")


if __name__ == '__main__':
    import sys
    
    # 默认标注 Round 1
    round_num = 1
    if len(sys.argv) > 1:
        round_num = int(sys.argv[1])
    
    # 执行标注
    result = annotate_round(round_num)
    
    if result:
        show_annotation_summary(result)
