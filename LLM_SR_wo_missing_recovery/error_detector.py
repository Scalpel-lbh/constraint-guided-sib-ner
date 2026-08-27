# -*- coding: utf-8 -*-
"""
结构纠错检测器

检测 LLM 标注中的常见错误，生成约束式反馈供 LLM 自我修正

重要设计原则（约束式反馈）：
- 不给正确答案
- 不裁 span
- 只说违反了哪条定义
- 把决策权留给 LLM
"""

import re
from typing import List, Dict, Tuple


# ============= 标注指南条款定义 =============
GUIDELINE_RULES = {
    'MISSING_ENTITY_CONSISTENCY': """The term "{entity_text}" appears {actual_count} times in the abstract, but you only annotated it {extracted_count} times. Please carefully check the abstract and ensure ALL occurrences of "{entity_text}" are annotated (unless the context clearly indicates it shouldn't be).""",

    'MISSING_WHITELIST_ROLE': """The text explicitly mentions the role indicator "{term}", but it was entirely missed. According to the guideline, role indicators like "anode", "cathode", "positive electrode", or "negative electrode" MUST be annotated as ROLE.""",

    'MISSING_WHITELIST_STRUCTURE': """The text explicitly mentions the core structure term "{term}", but it was entirely missed. According to the guideline, specific crystal/phase structures like "layered", "spinel", "olivine", "NASICON", "tunnel", "Prussian blue", etc., MUST be annotated as STRUCTURE.""",

    'MISSING_WHITELIST_MODIFICATION': """The text explicitly mentions the modification/action term "{term}", but it was entirely missed. According to the guideline, modification actions or chemical intervention terms such as "introduced" should be annotated as MODIFICATION when they describe how the material was modified.""",

    'MISSING_MATERIAL_CONTEXT': """A ROLE indicator ("{term}") was found in the abstract, but NO 'MATERIAL' entities were extracted at all. Usually, an anode/cathode refers to a specific core active material. Please review the context around "{term}" and ensure you haven't missed annotating the main active MATERIAL. (Note: do NOT annotate precursors, only the core active material).""",

    'STRUCTURE_SPAN': """STRUCTURE entities should only include the crystal/phase structure name itself (e.g., "P2", "O3", "NASICON", "layered", "spinel").
Generic words like "phase", "structure", or "type" should NOT be combined with structure codes (e.g., "P2 phase" and "P2-type" are invalid, should be just "P2").""",

    'STRUCTURE_COMBO_SUFFIX': """When annotating phase combinations like "P2/O3", only the structure identifiers should be included.
Descriptive words like "biphasic", "intergrown", "composite", or "hybrid" are NOT part of the STRUCTURE entity.""",

    'PHASE_TRANSITION': """Phase transition expressions (e.g., "P2-O2", "O3-P3") describe dynamic processes, NOT static crystal structures.
According to the annotation guideline, phase transitions should NOT be annotated as STRUCTURE entities.""",

    'PHASE_TRANSITION_SPLIT': """When a phase transition expression (e.g., "P2-O2", "P2-O3", "O3-P3") appears in text, it describes a dynamic process and should NOT be annotated at all.
Do NOT split and annotate the individual phases (e.g., "P2" and "O2") as separate STRUCTURE entities.
The entire transition expression should be left unannotated. You should DELETE these entities from your annotation.""",

    'PHASE_TRANSITION_KEYWORD': """Any text containing phase transition keywords (e.g., "phase transition", "transition", "transformation") describes a dynamic process, NOT a static structure.
Examples of invalid annotations: "P2-P2 phase transition", "P2-O2 transition", "phase transformation".
These expressions should NOT be annotated as STRUCTURE entities.""",

    'MATERIAL_PREFIX': """MATERIAL entities should only include the active material itself.
Dimensional descriptors (e.g., "3D", "2D", "1D") or morphological prefixes (e.g., "single-layer", "multi-layer") should NOT be part of the MATERIAL span.""",

    'MATERIAL_STRUCTURE_SPLIT': """When a structure descriptor (e.g., "disordered", "layered", "amorphous") appears before a material name, they should be annotated as SEPARATE entities.
The structure word should be STRUCTURE, and the material name should be MATERIAL.""",

    'TYPE_CONFLICT': """Each term has a defined entity type according to the annotation guideline.
Structure-related terms (e.g., "layered", "spinel") should be STRUCTURE.
Modification-related terms (e.g., "doped", "coated") should be MODIFICATION.
Role-related terms (e.g., "anode", "cathode") should be ROLE.""",

    'MATERIAL_SUFFIX': """MATERIAL entities should only include the material name/formula itself.
Generic suffixes like "material", "particles", "powder", "electrode", or "sample" should NOT be included in the MATERIAL span.""",

    'GENERIC_MATERIAL_TERM': """Generic material category terms (e.g., "layered oxides") are not specific materials and should NOT be annotated as MATERIAL.
Only annotate the structure term itself (e.g., "layered") as STRUCTURE when applicable.""",

    'ROLE_SUFFIX': """ROLE entities should be one of: "anode", "cathode", "positive electrode", "negative electrode".
When "anode material" or "cathode material" appears, only "anode" or "cathode" should be annotated as ROLE.
Note: "positive electrode" and "negative electrode" are VALID complete ROLE entities.""",

    'MORPHOLOGY_NOT_STRUCTURE': """Morphology descriptors (e.g., "nanoparticles", "nanosheets", "porous", "hollow", "hierarchical") describe physical shape, NOT crystal structure.
These terms should NOT be annotated as STRUCTURE entities.""",

    'FRAMEWORK_SPECIFICITY': """Only well-established crystallographic framework structures should be annotated as STRUCTURE.
Generic framework descriptions (e.g., "carbon framework", "3D framework", "conductive framework") are NOT specific crystal structures and should NOT be annotated as STRUCTURE.
Valid examples: "Prussian blue framework", "NASICON".""",

    'STRUCTURE_GENERIC_SUFFIX': """STRUCTURE entities should only include the structure term itself.
Generic words like "oxide", "compound", "material", "phase", "structure" should NOT be combined with structure descriptors.
For example: "layered oxide" should be split - only "layered" is STRUCTURE.""",

    'STRUCTURE_COMBO_BIPHASIC': """When annotating structure combinations (e.g., "layer/tunnel", "layered-spinel"), only the structure identifiers should be included.
Descriptive suffixes like "biphasic", "intergrown", "composite", "hybrid" are NOT part of the STRUCTURE entity.
For example: "layer/tunnel biphasic" should be just "layer" and "tunnel" separately, or at most "layer/tunnel".""",

    'MODIFICATION_COMPOUND_SPLIT': """When multiple modifications appear together (e.g., "N-doped carbon coated"), they should be annotated as SEPARATE MODIFICATION entities.
Each modification term (e.g., "N-doped", "carbon coated", "Fe-doped") should be its own MODIFICATION entity.
Do NOT combine multiple modifications into a single entity.""",

    'MATERIAL_MORPHOLOGY_PREFIX': """MATERIAL entities should only include the material name itself.
Morphology descriptors (e.g., "hollow", "porous", "mesoporous", "hierarchical", "spherical") should NOT be part of the MATERIAL span.
For example: "hollow nanocarbon" should be just "nanocarbon"; "porous carbon" should be just "carbon".""",

    'MATERIAL_MODIFICATION_SPLIT': """When a MATERIAL entity contains modification keywords (e.g., "doped", "coated", "decorated"), it indicates a composite expression that MUST be split into separate entities:
1. The modification part (e.g., "Fe-doped", "carbon coated") should be annotated as MODIFICATION.
2. The material name (e.g., "graphene", "NaTi2(PO4)3") should be annotated as MATERIAL.
Both parts MUST be retained - do not delete either the modification or the material name.""",

    'PHASE_MATERIAL_SPLIT': """When a phase code (P2, P3, O2, O3) is connected by a hyphen to a material name or chemical formula, they MUST be split into TWO separate entities:
1. The phase code (e.g., P2, O3) should be annotated as STRUCTURE.
2. The material name/formula (e.g., NNM, Na0.6MnO2) should be annotated as MATERIAL.
Both parts MUST be retained - do not delete either the phase code or the material name.""",

    'MATERIAL_ABBREVIATION_SPLIT': """When a material full name and its abbreviation are both written together, they should be annotated as SEPARATE MATERIAL entities.
For example, "hard carbon (HC)", "Na3V2(PO4)3(NVP)", and "carbon nanotubes (CNTs)" should not be kept as one span.
Annotate the full material name and the abbreviation as independent MATERIAL entities.""",

    'DESCRIPTIVE_WORD_NOT_STRUCTURE': """Descriptive words (e.g., "biphasic", "intergrown", "composite", "hybrid") describe the relationship or combination of structures, NOT the structures themselves.
These terms should NOT be annotated as STRUCTURE entities when they appear alone.""",

    'ROLE_WHITELIST': """ROLE entities must be one of the following: "anode", "cathode", "positive electrode", "negative electrode".
Any other terms should NOT be annotated as ROLE.
For example: "SIBs", "batteries", "sodium-ion batteries" are NOT valid ROLE entities.""",

    'NON_ENTITY_TERM': """The term "{term}" is a descriptive label or shorthand used in literature, not a named entity in this task.
It should NOT be annotated as any entity type. Please remove it from the annotation list.""",
}


class ErrorDetector:
    """
    检测 LLM 标注中的结构性错误
    """
    
    def __init__(self):
        # ============= 规则1: 相位标注应该简洁 =============
        # P2 phase, O3 phase 等应该只标注 P2, O3
        # 注意: P2-type / P2 type / NASICON-type 都应去掉 type 后缀
        self.phase_patterns = [
            # "P2 phase", "O3 phase", "P2 structure" 等是错误的（空格分隔 + phase/structure）
            (r'^(P2|P3|O2|O3)\s+(phase|structure)$', 'STRUCTURE'),
            # 新规范: 含 type 后缀的结构实体应仅保留核心结构词
            (r'^(P2|P3|O2|O3|NASICON)\s*[-–—]\s*type$', 'STRUCTURE'),
            (r'^(P2|P3|O2|O3|NASICON)\s+type$', 'STRUCTURE'),
        ]
        
        # ============= 规则1b: 相位组合标注不应包含描述词 =============
        # P2/O3 biphasic, P2-O3 intergrown 等应该只标注 P2/O3 或分别标注
        self.core_structure_terms = [
            'P2', 'P3', 'O2', 'O3', 'NASICON', 'spinel', 'olivine', 'layered', 'tunnel'
        ]
        self.structure_modifier_suffixes = [
            'type', 'structured', 'like'
        ]

        self.phase_combo_suffixes = [
            'biphasic', 'intergrown', 'composite', 'hybrid', 
            'mixture', 'heterostructure', 'integration'
        ]
        
        # ============= 规则2: 动态相变表示不应标注 =============
        # P2-O2, P2-P3, O3-P3 等表示相变过程，不标注
        self.phase_transition_pattern = r'^(P2|P3|O2|O3)[-–—](P2|P3|O2|O3)$'
        
        # ============= 规则2c: 相变关键词检测 =============
        # 包含 phase transition, transformation 等关键词的不应标为 STRUCTURE
        self.phase_transition_keywords = [
            'phase transition', 'phase transitions',
            'transition', 'transitions',
            'transformation', 'transformations',
            'phase change', 'phase changes'
        ]
        # 更一般的相位代码模式，用于识别 P2-OP4 transition 这类动态相变表达。
        # 保留上面的窄模式定义以兼容旧逻辑，这里在后面覆盖为更宽的匹配。
        self.phase_like_token_pattern = r"(?:P|O|OP)[\"'′]?\d+"
        self.phase_transition_pattern = (
            rf'^(?P<left>{self.phase_like_token_pattern})[-–—](?P<right>{self.phase_like_token_pattern})$'
        )
        
        # ============= 规则3: 不应包含的前缀 =============
        # 3D network, single layer, multi-layer 等不应作为 MATERIAL 的一部分
        self.invalid_prefixes = [
            '3D network',
            '3D',
            'single layer',
            'single-layer', 
            'multi-layer',
            'multi layer',
            'double layer',
            'double-layer',
            '2D',
            '1D',
        ]
        
        # ============= 规则4: 复合实体应拆分 =============
        # "disordered graphene" 应拆分为 disordered(STRUCTURE) + graphene(MATERIAL)
        # 注意：只有真正的晶体结构词才需要拆分，形貌词不算
        self.structure_prefixes = [
            'disordered',
            'ordered',
            'amorphous',
            'crystalline',
            'layered',
            # 注意：porous, hierarchical 等形貌词不在此列表，它们应该标为 O
        ]
        
        # ============= 规则6: MATERIAL 不应包含的后缀 =============
        # Na3V2(PO4)3 particles -> 只标 Na3V2(PO4)3
        self.material_invalid_suffixes = [
            'material', 'materials',
            'electrode', 'electrodes', 
            'particles', 'particle',
            'powder', 'powders',
            'sample', 'samples',
            'product', 'products',
        ]
        
        # ============= 规则7: ROLE 不应包含的后缀 =============
        # anode material -> 只标 anode
        # 注意: "positive electrode" 和 "negative electrode" 是完整的 ROLE，不应拆分
        # 只有 "anode material", "cathode electrode" 等才需要去掉后缀
        self.role_invalid_suffixes = [
            'material', 'materials',
        ]
        # 额外规则：anode electrode / cathode electrode 应该只保留 anode / cathode
        self.role_redundant_patterns = [
            (r'^(anode|cathode)\s+electrode$', 'ROLE'),
        ]
        
        # ============= 规则8: 形貌词不应标为 STRUCTURE =============
        self.morphology_words = [
            'nanoparticles', 'nanoparticle',
            'nanorods', 'nanorod',
            'nanosheets', 'nanosheet',
            'nanowires', 'nanowire',
            'nanotubes', 'nanotube',
            'nanofibers', 'nanofiber',
            'microspheres', 'microsphere',
            'microparticles', 'microparticle',
            'hollow',
            'porous',
            'mesoporous',
            'microporous',
            'hierarchical',
        ]
        
        # ============= 规则9: 泛化 framework 不应标为 STRUCTURE =============
        # 只有 Prussian blue framework 等专有结构才能标
        self.invalid_framework_prefixes = [
            'carbon', '3D', '3d', 'conductive', 'open', 'porous',
            'hierarchical', 'interconnected', 'continuous'
        ]
        
        # ============= 规则10: STRUCTURE 不应包含泛指词后缀 =============
        # layered oxide -> 只标 layered; P2 phase -> 只标 P2
        self.structure_invalid_suffixes = [
            'oxide', 'oxides',
            'compound', 'compounds', 
            'phase', 'phases',
            'structure', 'structures',
            'material', 'materials',
            'type', 'structured', 'like',
        ]
        
        # ============= 规则11: 结构组合 + biphasic 等后缀 =============
        # layer/tunnel biphasic, layered-spinel composite 等
        self.structure_combo_patterns = [
            r'^(layer|layered|tunnel|spinel|olivine|NASICON)[/\-](layer|layered|tunnel|spinel|olivine|NASICON)\s+(biphasic|intergrown|composite|hybrid|mixture)$',
        ]
        
        # ============= 规则12: MODIFICATION 复合词应拆分 =============
        # "N-doped carbon coated" 应拆分为 "N-doped" 和 "carbon coated"
        # 匹配模式：包含多个修饰词的组合
        self.modification_keywords = [
            'doped', 'doping', 'coated', 'coating', 'decorated', 'modified',
            'wrapped', 'encapsulated', 'embedded', 'anchored', 'loaded',
            'functionalized', 'grafted', 'intercalated', 'substituted', 'introduced'
        ]
        
        # ============= 规则13: MATERIAL 形貌前缀应去除 =============
        # "hollow nanocarbon" 应只标注 "nanocarbon"
        self.material_morphology_prefixes = [
            'hollow', 'porous', 'mesoporous', 'microporous', 'macroporous',
            'hierarchical', 'spherical', 'tubular', 'fibrous', 'flower-like',
            'core-shell', 'yolk-shell', 'sheet-like', 'rod-like', 'wire-like',
            'ultrathin', 'thick', 'thin', 'dense', 'loose',
        ]
        
        # ============= 规则14: MATERIAL 中包含 MODIFICATION 关键词应拆分 =============
        # "graphene coated NVTPF" 应拆分为 graphene + coated + NVTPF
        # 复用规则12的modification_keywords列表
        
        # ============= 规则15: 相位代码-材料名应拆分 =============
        # "P2-NNM", "O3-Na...", "P2/O3-Na..." 应拆分为 相位(STRUCTURE) + 材料名(MATERIAL)
        # 修复: 允许连续多相组合，如 P2/O3
        self.phase_material_pattern = r'^((?:P2|P3|O2|O3)(?:/(?:P2|P3|O2|O3))*)[-–—]([A-Z][A-Za-z0-9.()\s]+)$'
        
        # ============= 规则16: 描述性词汇不应标为 STRUCTURE =============
        # biphasic, intergrown 等只是描述词，不是结构本身
        self.phase_material_pattern = r'^((?:P2|P3|O2|O3)(?:/(?:P2|P3|O2|O3))*)[-–—](?=[A-Z])(.+)$'

        phase_code_pattern = r'(?:P-?2|P-?3|O-?2|O-?3)'
        self.phase_material_pattern = rf'^((?:{phase_code_pattern})(?:/(?:{phase_code_pattern}))*)[-–—](?=[A-Z])(.+)$'

        self.descriptive_words = [
            'biphasic', 'intergrown', 'composite', 'hybrid',
            'mixture', 'heterostructure', 'integration'
        ]

        # ============= 规则16b: 泛化材料类词不应标为 MATERIAL =============
        # 如 "layered oxides" 只应标注 layered 为 STRUCTURE
        self.generic_material_patterns = [
            r'^(layered|spinel|olivine|nasicon|tunnel)\s+oxides?$',
        ]
        
        # ============= 规则17: ROLE 白名单检查 =============
        # 只有这些词可以标为 ROLE（包括复数形式）
        self.valid_roles = [
            'anode', 'anodes',
            'cathode', 'cathodes', 
            'positive electrode', 'positive electrodes',
            'negative electrode', 'negative electrodes'
        ]

        self.modification_whitelist = [
            'introduced',
        ]

        # ============= 规则18: 明确不标注的描述性术语 =============
        # high-entropy / HEOs 等是描述词或缩写，不作为实体
        self.non_entity_terms = {
            'high-entropy',
            'high entropy',
            'heo',
            'heos',
        }
        
        # ============= 规则5: 类型冲突检测 =============
        # 某些词汇有固定的标签类型
        self.type_rules = {
            # MODIFICATION 相关
            'doped': 'MODIFICATION',
            'doping': 'MODIFICATION',
            'co-doped': 'MODIFICATION',
            'coated': 'MODIFICATION',
            'coating': 'MODIFICATION',
            'modified': 'MODIFICATION',
            'substituted': 'MODIFICATION',
            'introduced': 'MODIFICATION',
            'functionalized': 'MODIFICATION',
            'oxygen-vacancy': 'MODIFICATION',
            'defect-rich': 'MODIFICATION',
            'vacancy': 'MODIFICATION',
            
            # STRUCTURE 相关（仅限真正的晶体/相结构）
            'layered': 'STRUCTURE',
            'spinel': 'STRUCTURE',
            'olivine': 'STRUCTURE',
            'NASICON': 'STRUCTURE',
            'amorphous': 'STRUCTURE',
            'crystalline': 'STRUCTURE',
            'disordered': 'STRUCTURE',
            'ordered': 'STRUCTURE',
            'tunnel': 'STRUCTURE',
            'tunnel structure': 'STRUCTURE',
            
            # ROLE 相关
            'cathode': 'ROLE',
            'anode': 'ROLE',
            'positive electrode': 'ROLE',
            'negative electrode': 'ROLE',
        }
        # Normalize keys to lowercase so case variants in annotations are consistently checked.
        self.type_rules = {k.lower(): v for k, v in self.type_rules.items()}
        
    def detect_errors(self, entities: List[Dict], abstract: str) -> List[Dict]:
        """
        检测标注中的错误
        
        Args:
            entities: LLM 标注的实体列表 [{"text": ..., "label": ..., "context": ...}, ...]
            abstract: 原始摘要文本
            
        Returns:
            errors: 错误列表，每个错误包含 entity_index 用于精确定位
        """
        errors = []
        
        # 为了防止漏标探针和复合词拆分规则冲突（导致大模型幻觉），先收集所有实体的规则级错误
        rule_errors = []
        for idx, entity in enumerate(entities):
            text = entity.get('text', '')
            label = entity.get('label', '')

            # 规则18: 明确不标注的描述性术语
            error = self._check_non_entity_term(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则1: 检查相位标注是否过长
            error = self._check_phase_annotation(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则1b: 检查相位组合是否包含不应有的后缀
            error = self._check_phase_combo_suffix(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则2: 检查是否是相变表示
            error = self._check_phase_transition(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则2b: 检查是否是相变表达式的拆分标注
            error = self._check_phase_transition_split(text, label, entity.get('context', ''), abstract)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则2c: 检查是否包含相变关键词
            error = self._check_phase_transition_keyword(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则3: 检查是否包含无效前缀
            error = self._check_invalid_prefix(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue

            # 规则3b: 检查是否为泛化材料类词（如 layered oxides）
            error = self._check_generic_material_term(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则4: 检查是否需要拆分
            error = self._check_need_split(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则5: 检查类型冲突
            error = self._check_type_conflict(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则6: 检查 MATERIAL 是否包含无效后缀
            error = self._check_material_suffix(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则7: 检查 ROLE 是否包含无效后缀
            error = self._check_role_suffix(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则8: 检查形貌词是否被错误标为 STRUCTURE
            error = self._check_morphology_as_structure(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则9: 检查泛化 framework 是否被错误标为 STRUCTURE
            error = self._check_invalid_framework(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则10: 检查 STRUCTURE 是否包含泛指词后缀（如 layered oxide）
            error = self._check_structure_generic_suffix(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则11: 检查结构组合是否包含 biphasic 等后缀（如 layer/tunnel biphasic）
            error = self._check_structure_combo_biphasic(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则12: 检查 MODIFICATION 复合词是否需要拆分
            error = self._check_modification_compound(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则13: 检查 MATERIAL 是否包含形貌前缀
            error = self._check_material_morphology_prefix(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则14: 检查 MATERIAL 是否包含修饰关键词（应拆分）
            error = self._check_material_abbreviation_split(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue

            error = self._check_material_modification_split(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则15: 检查是否是"相位代码-材料名"模式（应拆分）
            error = self._check_phase_material_split(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则16: 检查描述性词汇是否被错误标为 STRUCTURE
            error = self._check_descriptive_word_as_structure(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
            
            # 规则17: 检查ROLE是否在白名单中
            error = self._check_role_whitelist(text, label)
            if error:
                error['entity_index'] = idx
                rule_errors.append(error)
                continue
                
        errors.extend(rule_errors)
        
        # --- 步骤 2: 查漏检测 (Missing Entity Detection) ---
        # 传递已经查出的错误，防止对错误的连词（如 "N-doped carbon"）执行数量一致性检查
        # Ablation: w/o missing-entity recovery.
        # Do not generate global missing-entity feedback (entity_index == -1).
        missing_errors = []
        # 将漏标错误放到最前面，大模型优先看漏标
        errors = missing_errors + errors
        
        return errors
    
    def _count_occurrences_in_text(self, text: str, target: str) -> int:
        """安全地统计文本中某个词出现的次数（尽量避免成为其他词的子串，但保持宽容）"""
        if not target or not isinstance(target, str):
            return 0
        try:
            pattern = re.escape(target)
            # 左边界：如果目标词以字母或数字开头，要求它左边不能是字母或数字
            if target[0].isalnum():
                pattern = r'(?<![a-zA-Z0-9])' + pattern
            # 右边界：如果目标词以字母数字结尾，要求它右边不能有字母或数字
            # 对于像 Na3V2(PO4)3 或者 O3 这样的情况，边界很安全
            if target[-1].isalnum():
                pattern = pattern + r'(?![a-zA-Z0-9])'
            
            return len(re.findall(pattern, text))
        except Exception:
            # 回退策略：如果正则出错，直接用 count，尽管可能不准
            return text.count(target)

    def _check_missing_entities(self, entities: List[Dict], abstract: str, existing_errors: List[Dict] = None) -> List[Dict]:
        """查漏检测策略：维度1 (一致性宽容检测) + 维度2 (白名单检测)"""
        errors = []
        abstract_lower = abstract.lower()
        extracted_texts = [e.get('text', '') for e in entities if e.get('text')]
        extracted_texts_lower = [t.lower() for t in extracted_texts]
        
        # 将已经被其他规则（如复合适体拆分等）检出报错的文本加入黑名单，防止其触发一致性漏标
        flawed_texts = set()
        if existing_errors:
            for err in existing_errors:
                if 'entity' in err:
                    flawed_texts.add(err['entity'])

        # ---------------- 维度1：一致性查漏 (Substring Tolerant Consistency) ----------------
        # 统计 JSON 里已经包含的独立词，看看是否漏掉了文中的其它实例
        unique_extracted_texts = list(set(extracted_texts))
        for text in unique_extracted_texts:
            if len(text) < 2:  # 忽略太短的词汇（例如单字母），容易误报
                continue
                
            # 如果这个词本身就被标错了（比如是个应该被拆分的复合词如 "N-doped carbon"），就不再要求模型去"补齐"它
            if text in flawed_texts:
                continue
                
            # 文中真实出现次数
            actual_count = self._count_occurrences_in_text(abstract, text)
            if actual_count > 0:
                # 宽容统计：只要文中的实体串包含了这个词，就算作 1 次提取（比如 P2 存在于 P2-Na... 中）
                extracted_count = sum(1 for ext_text in extracted_texts if text in ext_text)
                
                # 如果发现原文次数多于提取次数，并且差距不合理，触发一致性漏标
                if extracted_count < actual_count:
                    errors.append({
                        'error_type': 'MISSING_ENTITY_CONSISTENCY',
                        'entity': text,
                        'entity_type': 'MULTIPLE', # 可能被标错了类型，所以不纠结
                        'guideline_key': 'MISSING_ENTITY_CONSISTENCY',
                        'format_kwargs': {
                            'entity_text': text,
                            'actual_count': actual_count,
                            'extracted_count': extracted_count
                        },
                        'entity_index': -1 # 这是一个全局级错误，不绑定到单个现有 entity
                    })

        # ---------------- 维度2：固化集白名单兜底 (ROLE & STRUCTURE) ----------------
        
        # 2.1 检查 ROLE 白名单
        role_whitelist = ['anode', 'cathode', 'positive electrode', 'negative electrode']
        found_role_in_abstract = None
        for role in role_whitelist:
            if role in abstract_lower:
                # 如果这个 ROLE 相关的词既没被提取，也不在任何提取出的文本子串里
                if not any(role in ext_t for ext_t in extracted_texts_lower):
                    errors.append({
                        'error_type': 'MISSING_WHITELIST_ROLE',
                        'entity': role,
                        'entity_type': 'ROLE',
                        'guideline_key': 'MISSING_WHITELIST_ROLE',
                        'format_kwargs': {'term': role},
                        'entity_index': -1
                    })
                # 记录找到的 ROLE（为了下面启发式判断 MATERIAL）
                if found_role_in_abstract is None:
                    found_role_in_abstract = role
                    
        # 2.2 检查 STRUCTURE 白名单
        structure_whitelist = ['layered', 'spinel', 'olivine', 'nasicon', 'tunnel', 'prussian blue']
        for struct in structure_whitelist:
            if struct in abstract_lower:
                if not any(struct in ext_t for ext_t in extracted_texts_lower):
                    # 额外的一层安全网：比如 "tunnel" 不能是在 "tunneling electron" 里，简单用单词边界卡一下
                    if re.search(r'\b' + re.escape(struct) + r'\b', abstract_lower):
                        errors.append({
                            'error_type': 'MISSING_WHITELIST_STRUCTURE',
                            'entity': struct,
                            'entity_type': 'STRUCTURE',
                            'guideline_key': 'MISSING_WHITELIST_STRUCTURE',
                            'format_kwargs': {'term': struct},
                            'entity_index': -1
                        })

        # ---------------- 启发式探测：MATERIAL 探针 ----------------
        # 如果文章明确提到了某个极 (anode/cathode) 但是整个列表里一个 MATERIAL 都没有
        for mod in self.modification_whitelist:
            if re.search(r'\b' + re.escape(mod) + r'\b', abstract_lower):
                if not any(mod in ext_t for ext_t in extracted_texts_lower):
                    errors.append({
                        'error_type': 'MISSING_WHITELIST_MODIFICATION',
                        'entity': mod,
                        'entity_type': 'MODIFICATION',
                        'guideline_key': 'MISSING_WHITELIST_MODIFICATION',
                        'format_kwargs': {'term': mod},
                        'entity_index': -1
                    })

        if found_role_in_abstract is not None:
            has_material = any(e.get('label') == 'MATERIAL' for e in entities)
            if not has_material:
                errors.append({
                    'error_type': 'MISSING_MATERIAL_CONTEXT',
                    'entity': found_role_in_abstract,
                    'entity_type': 'MATERIAL',
                    'guideline_key': 'MISSING_MATERIAL_CONTEXT',
                    'format_kwargs': {'term': found_role_in_abstract},
                    'entity_index': -1
                })

        return errors

    def _check_phase_annotation(self, text: str, label: str) -> Dict:
        """规则1: 相位标注应该简洁"""
        for pattern, expected_label in self.phase_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return {
                    'error_type': 'STRUCTURE_SPAN_VIOLATION',
                    'entity': text,
                    'entity_type': label,
                    'guideline_key': 'STRUCTURE_SPAN',
                }
        if label == 'STRUCTURE':
            term_pattern = '|'.join(re.escape(t) for t in self.core_structure_terms)
            suffix_pattern = '|'.join(re.escape(s) for s in self.structure_modifier_suffixes)
            if re.match(rf'^({term_pattern})\s*[-–—]\s*({suffix_pattern})$', text, re.IGNORECASE):
                return {
                    'error_type': 'STRUCTURE_SPAN_VIOLATION',
                    'entity': text,
                    'entity_type': label,
                    'guideline_key': 'STRUCTURE_SPAN',
                }
        return None

    def _check_non_entity_term(self, text: str, label: str) -> Dict:
        """规则18: 明确不标注的描述性术语"""
        if not text:
            return None

        text_norm = text.strip().lower()
        if text_norm in self.non_entity_terms:
            return {
                'error_type': 'NON_ENTITY_TERM_VIOLATION',
                'entity': text,
                'entity_type': label,
                'guideline_key': 'NON_ENTITY_TERM',
                'format_kwargs': {'term': text},
            }

        return None
    
    def _check_phase_combo_suffix(self, text: str, label: str) -> Dict:
        """规则1b: 相位组合标注不应包含描述词后缀"""
        if label != 'STRUCTURE':
            return None
        
        combo_pattern = r'^((?:P2|P3|O2|O3)[/\-](?:P2|P3|O2|O3))\s+(\w+)$'
        match = re.match(combo_pattern, text, re.IGNORECASE)
        
        if match:
            suffix = match.group(2)
            if suffix.lower() in [s.lower() for s in self.phase_combo_suffixes]:
                return {
                    'error_type': 'STRUCTURE_COMBO_SUFFIX_VIOLATION',
                    'entity': text,
                    'entity_type': label,
                    'guideline_key': 'STRUCTURE_COMBO_SUFFIX',
                }
        return None
    
    def _check_phase_transition(self, text: str, label: str) -> Dict:
        """规则2: 相变表示不应标注（包括同相转变如P2-P2）"""
        # 检查标准相变模式（P2-O2等）
        if re.match(self.phase_transition_pattern, text, re.IGNORECASE):
            return {
                'error_type': 'PHASE_TRANSITION_VIOLATION',
                'entity': text,
                'entity_type': label,
                'guideline_key': 'PHASE_TRANSITION',
            }
        
        # 检查同相"转变"（如P2-P2, O3-O3等，这没有意义）
        same_phase_pattern = r'^(P2|P3|O2|O3)[-–—]\1\b'
        if re.match(same_phase_pattern, text, re.IGNORECASE):
            return {
                'error_type': 'PHASE_TRANSITION_VIOLATION',
                'entity': text,
                'entity_type': label,
                'guideline_key': 'PHASE_TRANSITION',
            }
        
        return None
    
    def _is_phase_like_token(self, text: str) -> bool:
        """判断一个实体文本是否像独立的相位代码，如 P2 / O3 / OP4。"""
        return bool(re.match(rf'^{self.phase_like_token_pattern}$', text.strip(), re.IGNORECASE))

    def _contains_phase_transition_keyword(self, text: str) -> bool:
        """判断上下文是否明确在描述动态相变过程。"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.phase_transition_keywords)

    def _check_phase_transition_split(self, text: str, label: str, context: str, abstract: str) -> Dict:
        """规则2b: 检查是否是相变表达式的拆分标注
        
        例如：原文中有 "P2-O2 transition"，LLM 不应该把 "P2" 或 "O2" 单独标注为 STRUCTURE
        """
        if label != 'STRUCTURE':
            return None
        
        # 只检查单独的相位代码
        text_upper = text.upper().strip()

        if not self._is_phase_like_token(text_upper):
            return None
        
        # Prefer context-level detection to avoid abstract-wide false positives.
        search_text = context if context else ''
        if not search_text:
            return None

        if not self._contains_phase_transition_keyword(search_text):
            return None

        code_pattern = re.compile(rf'\b{re.escape(text_upper)}\b', re.IGNORECASE)
        matches = list(code_pattern.finditer(search_text))
        if not matches:
            return None

        # Flag only when every occurrence of this phase code in context is part of a transition.
        has_transition_occurrence = False
        for m in matches:
            start, end = m.start(), m.end()
            left = search_text[:start]
            right = search_text[end:]

            broad_right_transition = re.match(
                rf'^\s*[-–—]\s*{self.phase_like_token_pattern}\b',
                right,
                re.IGNORECASE
            )
            broad_left_transition = re.search(
                rf'\b{self.phase_like_token_pattern}\s*[-–—]\s*$',
                left,
                re.IGNORECASE
            )
            right_transition = re.match(r'^\s*[-–—]\s*(P2|P3|O2|O3)\b', right, re.IGNORECASE)
            left_transition = re.search(r'\b(P2|P3|O2|O3)\s*[-–—]\s*$', left, re.IGNORECASE)

            if broad_right_transition or broad_left_transition or right_transition or left_transition:
                has_transition_occurrence = True
            else:
                return None

        if has_transition_occurrence:
            return {
                'error_type': 'PHASE_TRANSITION_SPLIT_VIOLATION',
                'entity': text,
                'entity_type': label,
                'guideline_key': 'PHASE_TRANSITION_SPLIT',
            }
        
        return None
    
    def _check_phase_transition_keyword(self, text: str, label: str) -> Dict:
        """规则2c: 检查是否包含相变关键词（如 'P2-P2 phase transition'）"""
        if label != 'STRUCTURE':
            return None
        
        text_lower = text.lower()
        
        # 检查是否包含相变关键词
        for keyword in self.phase_transition_keywords:
            if keyword in text_lower:
                return {
                    'error_type': 'PHASE_TRANSITION_KEYWORD_VIOLATION',
                    'entity': text,
                    'entity_type': label,
                    'guideline_key': 'PHASE_TRANSITION_KEYWORD',
                }
        
        return None
    
    def _check_invalid_prefix(self, text: str, label: str) -> Dict:
        """规则3: 检查无效前缀"""
        if label != 'MATERIAL':
            return None
            
        text_lower = text.lower()
        for prefix in self.invalid_prefixes:
            prefix_lower = prefix.lower()
            if text_lower.startswith(prefix_lower + ' ') or text_lower.startswith(prefix_lower + '-'):
                return {
                    'error_type': 'MATERIAL_PREFIX_VIOLATION',
                    'entity': text,
                    'entity_type': label,
                    'guideline_key': 'MATERIAL_PREFIX',
                }
        return None
    
    def _check_need_split(self, text: str, label: str) -> Dict:
        """规则4: 检查是否需要拆分为多个实体"""
        if label != 'MATERIAL':
            return None
            
        text_lower = text.lower()
        for struct_prefix in self.structure_prefixes:
            prefix_lower = struct_prefix.lower()
            if text_lower.startswith(prefix_lower + ' ') or text_lower.startswith(prefix_lower + '-'):
                remaining = text[len(struct_prefix):].lstrip(' -')
                if remaining and len(remaining) > 2:
                    return {
                        'error_type': 'MATERIAL_STRUCTURE_SPLIT_VIOLATION',
                        'entity': text,
                        'entity_type': label,
                        'guideline_key': 'MATERIAL_STRUCTURE_SPLIT',
                    }
        return None
    
    def _check_type_conflict(self, text: str, label: str) -> Dict:
        """规则5: 检查类型冲突"""
        text_lower = text.lower().strip()
        
        if text_lower in self.type_rules:
            expected_label = self.type_rules[text_lower]
            if label != expected_label:
                return {
                    'error_type': 'TYPE_CONFLICT_VIOLATION',
                    'entity': text,
                    'entity_type': label,
                    'expected_type': expected_label,
                    'guideline_key': 'TYPE_CONFLICT',
                }
        return None
    
    def _check_material_suffix(self, text: str, label: str) -> Dict:
        """规则6: MATERIAL 不应包含无效后缀"""
        if label != 'MATERIAL':
            return None
        
        text_lower = text.lower()
        for suffix in self.material_invalid_suffixes:
            suffix_lower = suffix.lower()
            if text_lower.endswith(' ' + suffix_lower):
                return {
                    'error_type': 'MATERIAL_SUFFIX_VIOLATION',
                    'entity': text,
                    'entity_type': label,
                    'guideline_key': 'MATERIAL_SUFFIX',
                }
        return None

    def _check_generic_material_term(self, text: str, label: str) -> Dict:
        """规则3b: 泛化材料类词不应标为 MATERIAL"""
        if label != 'MATERIAL':
            return None

        text_lower = text.lower().strip()
        for pattern in self.generic_material_patterns:
            if re.match(pattern, text_lower):
                return {
                    'error_type': 'GENERIC_MATERIAL_TERM_VIOLATION',
                    'entity': text,
                    'entity_type': label,
                    'guideline_key': 'GENERIC_MATERIAL_TERM',
                }

        return None
    
    def _check_role_suffix(self, text: str, label: str) -> Dict:
        """规则7: ROLE 不应包含无效后缀
        
        注意:
        - "positive electrode" 和 "negative electrode" 是完整的 ROLE，不报错
        - "anode material", "cathode material" 应该只标 anode/cathode
        - "anode electrode", "cathode electrode" 应该只标 anode/cathode
        """
        if label != 'ROLE':
            return None
        
        text_lower = text.lower()
        
        # 检查无效后缀 (material, materials)
        for suffix in self.role_invalid_suffixes:
            suffix_lower = suffix.lower()
            if text_lower.endswith(' ' + suffix_lower):
                return {
                    'error_type': 'ROLE_SUFFIX_VIOLATION',
                    'entity': text,
                    'entity_type': label,
                    'guideline_key': 'ROLE_SUFFIX',
                }
        
        # 检查冗余表达 (anode electrode, cathode electrode)
        for pattern, _ in self.role_redundant_patterns:
            if re.match(pattern, text_lower):
                return {
                    'error_type': 'ROLE_SUFFIX_VIOLATION',
                    'entity': text,
                    'entity_type': label,
                    'guideline_key': 'ROLE_SUFFIX',
                }
        
        return None
    
    def _check_morphology_as_structure(self, text: str, label: str) -> Dict:
        """规则8: 形貌词不应标为 STRUCTURE"""
        if label != 'STRUCTURE':
            return None
        
        text_lower = text.lower().strip()
        if text_lower in [m.lower() for m in self.morphology_words]:
            return {
                'error_type': 'MORPHOLOGY_AS_STRUCTURE_VIOLATION',
                'entity': text,
                'entity_type': label,
                'guideline_key': 'MORPHOLOGY_NOT_STRUCTURE',
            }
        return None
    
    def _check_invalid_framework(self, text: str, label: str) -> Dict:
        """规则9: 泛化 framework 不应标为 STRUCTURE"""
        if label != 'STRUCTURE':
            return None
        
        text_lower = text.lower()
        if 'framework' not in text_lower:
            return None
        
        for prefix in self.invalid_framework_prefixes:
            prefix_lower = prefix.lower()
            if text_lower.startswith(prefix_lower + ' ') or text_lower.startswith(prefix_lower + '-'):
                return {
                    'error_type': 'INVALID_FRAMEWORK_VIOLATION',
                    'entity': text,
                    'entity_type': label,
                    'guideline_key': 'FRAMEWORK_SPECIFICITY',
                }
        
        if text_lower in ['framework', 'open framework', 'open framework structure']:
            return {
                'error_type': 'INVALID_FRAMEWORK_VIOLATION',
                'entity': text,
                'entity_type': label,
                'guideline_key': 'FRAMEWORK_SPECIFICITY',
            }
        
        return None

    def _check_structure_generic_suffix(self, text: str, label: str) -> Dict:
        """规则10: STRUCTURE 不应包含泛指词后缀（如 layered oxide）"""
        if label != 'STRUCTURE':
            return None
        
        text_lower = text.lower()
        words = text_lower.split()
        
        if len(words) < 2:
            return None
        
        # 检查最后一个词是否是泛指后缀
        last_word = words[-1]
        if last_word in [s.lower() for s in self.structure_invalid_suffixes]:
            return {
                'error_type': 'STRUCTURE_GENERIC_SUFFIX_VIOLATION',
                'entity': text,
                'entity_type': label,
                'guideline_key': 'STRUCTURE_GENERIC_SUFFIX',
            }
        
        return None

    def _check_structure_combo_biphasic(self, text: str, label: str) -> Dict:
        """规则11: 结构组合不应包含 biphasic 等后缀（如 layer/tunnel biphasic）"""
        if label != 'STRUCTURE':
            return None
        
        for pattern in self.structure_combo_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return {
                    'error_type': 'STRUCTURE_COMBO_BIPHASIC_VIOLATION',
                    'entity': text,
                    'entity_type': label,
                    'guideline_key': 'STRUCTURE_COMBO_BIPHASIC',
                }
        
        return None

    def _check_modification_compound(self, text: str, label: str) -> Dict:
        """规则12: MODIFICATION 复合词应拆分（如 'N-doped carbon coated'）"""
        if label != 'MODIFICATION':
            return None
        
        text_lower = text.lower()
        
        # 统计文本中包含多少个修饰关键词
        keyword_count = 0
        for keyword in self.modification_keywords:
            # 使用单词边界匹配，避免部分匹配
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                keyword_count += 1
        
        # 如果包含2个或以上的修饰词，应该拆分
        if keyword_count >= 2:
            return {
                'error_type': 'MODIFICATION_COMPOUND_VIOLATION',
                'entity': text,
                'entity_type': label,
                'guideline_key': 'MODIFICATION_COMPOUND_SPLIT',
            }
        
        return None

    def _check_material_morphology_prefix(self, text: str, label: str) -> Dict:
        """规则13: MATERIAL 不应包含形貌前缀（如 'hollow nanocarbon'）"""
        if label != 'MATERIAL':
            return None
        
        text_lower = text.lower()
        
        # 检查是否以形貌前缀开头
        for prefix in self.material_morphology_prefixes:
            prefix_lower = prefix.lower()
            # 检查 "prefix " 或 "prefix-" 开头
            if text_lower.startswith(prefix_lower + ' ') or text_lower.startswith(prefix_lower + '-'):
                # 确保去掉前缀后还有内容（即真的是 "前缀 + 材料" 的结构）
                remaining = text_lower[len(prefix_lower):].lstrip(' -')
                if remaining:  # 还有剩余内容
                    return {
                        'error_type': 'MATERIAL_MORPHOLOGY_PREFIX_VIOLATION',
                        'entity': text,
                        'entity_type': label,
                        'guideline_key': 'MATERIAL_MORPHOLOGY_PREFIX',
                    }
        
        return None

    def _check_material_abbreviation_split(self, text: str, label: str) -> Dict:
        """Rule 13b: material full name and trailing abbreviation should be split."""
        if label != 'MATERIAL':
            return None

        text_stripped = text.strip()
        if '(' not in text_stripped or not text_stripped.endswith(')'):
            return None

        match = re.match(r'^(.+?)\s*\(([A-Za-z][A-Za-z0-9+-]{1,12}s?)\)$', text_stripped)
        if not match:
            return None

        full_name = match.group(1).strip()
        abbreviation = match.group(2).strip()
        if not full_name or not abbreviation:
            return None

        if re.fullmatch(r'[A-Z][a-z]?\d*O\d*', abbreviation):
            return None

        return {
            'error_type': 'MATERIAL_ABBREVIATION_SPLIT_VIOLATION',
            'entity': text,
            'entity_type': label,
            'guideline_key': 'MATERIAL_ABBREVIATION_SPLIT',
        }

    def _check_material_modification_split(self, text: str, label: str) -> Dict:
        """规则14: MATERIAL 中包含修饰关键词应拆分
        
        检测场景：
        1. "graphene coated NVTPF" → graphene + coated + NVTPF (3个实体)
        2. "carbon coated silicon" → carbon coated + silicon (2个实体)  
        3. "N-doped carbon" → N-doped + carbon (2个实体)
        
        核心逻辑：只要MATERIAL实体包含修饰关键词且不是单个词，就应该拆分
        """
        if label != 'MATERIAL':
            return None
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Single-token spans are usually skipped, except hyphenated forms like "Fe-doped"
        # which should still be split into modification + material-related parts.
        if len(words) < 2 and '-' not in text_lower:
            return None
        
        # 检查是否包含修饰关键词
        for keyword in self.modification_keywords:
            # 使用单词边界匹配
            pattern = r'\b' + re.escape(keyword) + r'\b'
            match = re.search(pattern, text_lower)
            
            if match:
                # 找到修饰词在文本中的位置
                keyword_position = match.start()
                
                # 检查修饰词前后是否都有内容
                before_text = text_lower[:keyword_position].strip()
                after_text = text_lower[match.end():].strip()
                
                # If there is material text before or after the modification token,
                # this MATERIAL span is a mixed expression and should be split.
                if before_text or after_text:
                    # 报错：包含修饰关键词的复合表达式不应整体标为MATERIAL
                    return {
                        'error_type': 'MATERIAL_MODIFICATION_SPLIT_VIOLATION',
                        'entity': text,
                        'entity_type': label,
                        'guideline_key': 'MATERIAL_MODIFICATION_SPLIT',
                    }
        
        return None
    
    def _check_role_whitelist(self, text: str, label: str) -> Dict:
        """规则17: ROLE必须在白名单中
        
        只有4个合法的ROLE：anode, cathode, positive electrode, negative electrode
        """
        if label != 'ROLE':
            return None
        
        text_lower = text.lower().strip()
        if text_lower not in [r.lower() for r in self.valid_roles]:
            return {
                'error_type': 'ROLE_WHITELIST_VIOLATION',
                'entity': text,
                'entity_type': label,
                'guideline_key': 'ROLE_WHITELIST',
            }
        return None
    
    def _check_descriptive_word_as_structure(self, text: str, label: str) -> Dict:
        """规则16: 描述性词汇不应标为 STRUCTURE
        
        biphasic, intergrown 等词描述结构之间的关系，不是结构本身
        """
        if label != 'STRUCTURE':
            return None
        
        text_lower = text.lower().strip()
        if text_lower in [w.lower() for w in self.descriptive_words]:
            return {
                'error_type': 'DESCRIPTIVE_WORD_AS_STRUCTURE_VIOLATION',
                'entity': text,
                'entity_type': label,
                'guideline_key': 'DESCRIPTIVE_WORD_NOT_STRUCTURE',
            }
        return None
    
    def _check_phase_material_split(self, text: str, label: str) -> Dict:
        """规则15: 相位代码-材料名应拆分（如 'P2-NNM' 标为MATERIAL是错误的）
        正确标注：P2(STRUCTURE) + 材料名(MATERIAL)
        """
        # 只检测MATERIAL和STRUCTURE标签
        if label not in ['MATERIAL', 'STRUCTURE']:
            return None
        
        # 检查是否匹配 "相位代码-材料名" 模式
        match = re.match(self.phase_material_pattern, text, re.IGNORECASE)
        if match and (
            self._is_phase_like_token(match.group(2).strip())
            or self._contains_phase_transition_keyword(text)
        ):
            return None
        
        if match:
            # 其他情况（材料名或化学式）整体标为MATERIAL或STRUCTURE都是错误的
            # 应该拆分为两个实体：P2(STRUCTURE) + 材料名(MATERIAL)
            return {
                'error_type': 'PHASE_MATERIAL_SPLIT_VIOLATION',
                'entity': text,
                'entity_type': label,
                'guideline_key': 'PHASE_MATERIAL_SPLIT',
            }
        
        return None

    def format_feedback(self, errors: List[Dict]) -> str:
        """
        将错误列表格式化为约束式反馈文本
        
        约束式反馈原则：
        - 不给正确答案
        - 不裁 span
        - 只说违反了哪条定义
        - 把决策权留给 LLM
        - 使用索引精确定位（避免同名实体混淆）
        
        Args:
            errors: 错误列表
            
        Returns:
            feedback: 格式化的约束式反馈文本
        """
        if not errors:
            return ""
        
        feedback_lines = [
            "The following annotation issues were detected. Please re-evaluate according to the annotation guidelines.",
            "IMPORTANT: Entity positions are specified by their index (0-based) in your annotation list.",
            ""
        ]
        
        for i, error in enumerate(errors, 1):
            entity = error.get('entity', '')
            entity_type = error.get('entity_type', '')
            entity_index = error.get('entity_index', '?')
            guideline_key = error.get('guideline_key', '')

            # 获取对应的 guideline 条款
            guideline_text = GUIDELINE_RULES.get(guideline_key, '')
            if 'format_kwargs' in error:
                try:
                    guideline_text = guideline_text.format(**error['format_kwargs'])
                except KeyError:
                    pass

            feedback_lines.append(f"[Issue {i}]")
            # 全局级别的漏标没有具体的 entity_index
            if entity_index == -1:
                feedback_lines.append(f"Global Document Issue: Missing Entity Detection")
            else:
                feedback_lines.append(f"Entity Index: #{entity_index}")  # 精确定位
            feedback_lines.append(f"Entity Text: \"{entity}\"")
            feedback_lines.append(f"Annotated Type: {entity_type}")
            feedback_lines.append(f"Error Type: {error.get('error_type', 'UNKNOWN')}")
            
            # 如果有期望类型（类型冲突的情况）
            if 'expected_type' in error:
                feedback_lines.append(f"Note: This term is typically associated with {error['expected_type']} entities.")
            
            feedback_lines.append("")
            feedback_lines.append("Guideline Violation / System Check:")
            feedback_lines.append(guideline_text)
            feedback_lines.append("")
            if entity_index == -1:
                feedback_lines.append("Action: Please carefully scan the abstract and APPEND the missing entities to your JSON array.")
            else:
                feedback_lines.append("Action: Please re-evaluate entity #{} according to the guideline above.".format(entity_index))
            feedback_lines.append("")
            feedback_lines.append("-" * 50)
            feedback_lines.append("")
        
        feedback_lines.append("Please output the complete corrected annotation result in JSON format.")
        feedback_lines.append("Keep your correct existing annotations unchanged. Follow the instructions to modify only the flagged entities.")
        return "\n".join(feedback_lines)
    
    def get_error_summary(self, errors: List[Dict]) -> Dict:
        """
        获取错误统计摘要
        """
        summary = {}
        for error in errors:
            error_type = error.get('error_type', 'UNKNOWN')
            summary[error_type] = summary.get(error_type, 0) + 1
        return summary


# 便捷函数
def detect_annotation_errors(entities: List[Dict], abstract: str) -> Tuple[List[Dict], str]:
    """
    检测标注错误并生成约束式反馈
    
    Returns:
        errors: 错误列表
        feedback: 格式化的约束式反馈文本
    """
    detector = ErrorDetector()
    errors = detector.detect_errors(entities, abstract)
    feedback = detector.format_feedback(errors)
    return errors, feedback


if __name__ == '__main__':
    # 测试约束式反馈
    detector = ErrorDetector()
    
    # 测试用例
    test_entities = [
        {"text": "P2 phase", "label": "STRUCTURE"},
        {"text": "P2/O3 biphasic", "label": "STRUCTURE"},
        {"text": "disordered graphene", "label": "MATERIAL"},
        {"text": "Na3V2(PO4)3 particles", "label": "MATERIAL"},
        {"text": "anode material", "label": "ROLE"},
        {"text": "porous", "label": "STRUCTURE"},
        {"text": "carbon framework", "label": "STRUCTURE"},
    ]
    
    print("=" * 60)
    print("约束式反馈测试")
    print("=" * 60)
    
    errors = detector.detect_errors(test_entities, "")
    
    print(f"\n检测到 {len(errors)} 个错误\n")
    
    print("格式化的约束式反馈:")
    print("-" * 60)
    feedback = detector.format_feedback(errors)
    print(feedback)
