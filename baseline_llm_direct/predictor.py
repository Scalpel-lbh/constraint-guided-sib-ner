# -*- coding: utf-8 -*-
"""
LLM 直接预测模块
- 读取 test_50.json（Label Studio 格式）
- 通过 DeepSeek + KNN few-shot 预测每条样本的实体
- 输出 output/predictions.json
"""

import json
import os
import sys
import time

from openai import OpenAI

# 将 LLM 追加到路径末尾（优先级低于本目录，避免覆盖本目录的 config）
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLM_DIR  = os.path.join(PROJECT_DIR, "LLM")
if LLM_DIR not in sys.path:
    sys.path.append(LLM_DIR)

# 导入 LLM 的对齐函数（不导入 llm_annotator，避免其 config 依赖）
from data_converter import align_predictions                   # type: ignore

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    TEST_FILE, TRAIN_FILE, OUTPUT_DIR,
    USE_KNN_FEWSHOT, KNN_K, KNN_MODEL,
)


# ============================================================
# Prompt 常量（与 LLM/llm_annotator.py 完全一致，内联以避免依赖）
# ============================================================
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


# ============================================================
# 轻量 KNN 检索器（自包含，不依赖 LLM config）
# ============================================================
import numpy as np

class _LocalKNNRetriever:
    """
    从 Label Studio 格式的训练数据中检索相似样本。
    完全自包含，不依赖 LLM 的 config.py。
    """
    def __init__(self, train_file: str, model_name: str, k: int):
        from sentence_transformers import SentenceTransformer
        self.k   = k
        print(f"加载 KNN embedding 模型: {model_name}")
        self.st  = SentenceTransformer(model_name)
        self.samples = self._load(train_file)
        texts = [s['text'] for s in self.samples]
        self.embeddings = self.st.encode(texts, show_progress_bar=False)
        print(f"KNN 检索库：{len(self.samples)} 条训练样本")

    def _load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        samples = []
        for item in data:
            text = item.get('data', {}).get('text', '')
            entities = []
            for anno in item.get('annotations', []):
                for r in anno.get('result', []):
                    if r.get('type') == 'labels':
                        v = r['value']
                        entities.append({
                            'text':  v.get('text', ''),
                            'label': v.get('labels', [''])[0],
                        })
            if text:
                samples.append({'text': text, 'entities': entities})
        return samples

    def retrieve(self, query: str):
        q_emb = self.st.encode([query], show_progress_bar=False)[0]
        sims  = np.dot(self.embeddings, q_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8
        )
        top_k = np.argsort(sims)[::-1][:self.k]
        return [
            {**self.samples[i], 'similarity': float(sims[i])}
            for i in top_k
        ]


_knn_retriever = None

def _get_knn_retriever():
    global _knn_retriever
    if _knn_retriever is None:
        _knn_retriever = _LocalKNNRetriever(TRAIN_FILE, KNN_MODEL, KNN_K)
    return _knn_retriever


# ============================================================
# Prompt 构建
# ============================================================

def build_prompt(abstract: str) -> str:
    prompt = BASE_SYSTEM_PROMPT + STATIC_FEWSHOT

    if USE_KNN_FEWSHOT:
        retriever = _get_knn_retriever()
        examples  = retriever.retrieve(abstract)
        knn_block = (
            "\n\n# Additional Examples (从人工标注训练集中检索)\n"
            "以下是与待标注文本最相似的人工标注示例，请重点参考这些标注风格：\n"
        )
        for i, ex in enumerate(examples, 1):
            text_preview  = ex['text'][:800] + "..." if len(ex['text']) > 800 else ex['text']
            entities_json = json.dumps(ex['entities'], ensure_ascii=False, indent=2)
            knn_block += (
                f"\n**检索示例 {i}** (相似度: {ex['similarity']:.3f})\n"
                f"Input: \"{text_preview}\"\nOutput:\n{entities_json}\n"
            )
        prompt += knn_block

    return prompt


# ============================================================
# 单条推理
# ============================================================

def _parse_response(raw: str):
    """解析 DeepSeek 返回的 JSON 字符串"""
    raw = raw.strip()
    # 去掉可能包裹的 markdown 代码块
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines).strip()
    try:
        return json.loads(raw)
    except Exception:
        return []


def predict_one(client: OpenAI, abstract: str, title: str = "", retry: int = 3):
    """
    对单条摘要调用 DeepSeek，返回 Label Studio 格式的标注结果列表。
    (同 align_predictions 返回的格式，可直接用于评估)
    """
    system_prompt = build_prompt(abstract)
    user_prompt   = f'Input: "{abstract}"\nOutput:'

    for attempt in range(retry):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=4096,
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = resp.choices[0].message.content
            entities = _parse_response(raw)
            if not isinstance(entities, list):
                entities = []
            # 用 align_predictions 将 {text, label, context} → 字符偏移
            ls_results = align_predictions(abstract, entities)
            return ls_results, entities, raw
        except Exception as e:
            print(f"  调用失败 (attempt {attempt+1}/{retry}): {e}")
            time.sleep(5 * (attempt + 1))

    return [], [], ""


# ============================================================
# 批量预测
# ============================================================

def predict_test_set(resume: bool = True):
    """
    预测整个测试集，输出 output/predictions.json。

    predictions.json 格式：
    [
      {
        "id": <LS id>,
        "title": "...",
        "abstract": "...",
        "ground_truth": [{"start":N, "end":M, "label":"..."},...],
        "prediction":   [{"start":N, "end":M, "label":"..."},...],
        "raw_entities": [...],   # LLM 原始输出（text/label/context）
      },
      ...
    ]
    """
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    predictions_file = os.path.join(OUTPUT_DIR, "predictions.json")

    # 断点续跑：加载已有结果
    done_ids = set()
    results  = []
    if resume and os.path.exists(predictions_file):
        with open(predictions_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        done_ids = {r['id'] for r in results}
        print(f"断点续跑：已完成 {len(done_ids)} / {len(test_data)} 条")

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    for idx, item in enumerate(test_data):
        item_id  = item.get('id')
        abstract = item.get('data', {}).get('text', '')
        title    = item.get('data', {}).get('title', '')

        if item_id in done_ids:
            continue

        # 提取 ground truth（字符偏移格式）
        gt_entities = []
        for anno in item.get('annotations', []):
            for r in anno.get('result', []):
                if r.get('type') == 'labels':
                    v = r['value']
                    gt_entities.append({
                        'start': v['start'],
                        'end':   v['end'],
                        'label': v['labels'][0],
                    })

        print(f"[{idx+1}/{len(test_data)}] {title[:60]}")
        ls_results, raw_entities, _ = predict_one(client, abstract, title)

        # ls_results 来自 align_predictions，格式: {"value":{start,end,text,labels},...}
        pred_entities = []
        for r in ls_results:
            v = r.get('value', {})
            pred_entities.append({
                'start': v['start'],
                'end':   v['end'],
                'label': v['labels'][0],
            })

        results.append({
            'id':           item_id,
            'title':        title,
            'abstract':     abstract,
            'ground_truth': gt_entities,
            'prediction':   pred_entities,
            'raw_entities': raw_entities,
        })
        done_ids.add(item_id)

        # 每条保存，防止中途崩溃
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 避免超出 API 速率
        time.sleep(1.0)

    print(f"\n预测完成，共 {len(results)} 条，结果已保存至 {predictions_file}")
    return results
