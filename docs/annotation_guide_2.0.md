# 🧾 钠离子电池正负极材料 NER 标注指南（修订版 2.0）

## 1. 任务目标

从文献摘要中抽取与**钠离子电池（Sodium-ion batteries, SIBs）正负极材料**相关的关键信息，用于构建材料领域命名实体识别模型。

本任务关注：
- 电极中的**核心活性材料及其组成形式**；
- 材料的**化学修饰方式**；
- 材料在电池中的**功能角色**；
- 材料的**晶体/相结构类型**。

---

## 2. 实体类型

本任务采用四类实体：

| 实体类型 | 含义 |
|----------|------|
| **MATERIAL** | 核心活性材料或其组成形式 |
| **MODIFICATION** | 对材料的化学修饰/改性方式 |
| **ROLE** | 材料在电池中的功能角色 |
| **STRUCTURE** | 晶体结构或相结构类型 |

---

## 3. MATERIAL（核心材料）

### 3.1 定义

表示在电极中起主要作用的**活性材料本体**，可以是：
- 单一材料；
- 明确的化学组成；
- 明确的组成/复合形式（如 X/C, X/graphene, X@C）。

👉 MATERIAL 回答“**材料是什么**”。

---

### 3.2 应标注为 MATERIAL

#### （1）化学式或材料名称
- `Na3V2(PO4)3`
- `Na0.67MnO2`
- `NaFePO4`
- `SnS2`
- `hard carbon`
- `graphite`
- `MXene`
- `Prussian blue`

#### （2）明确的组成/复合形式（整体标注）
- `Na3V2(PO4)3/C`
- `SnS2/graphene`
- `Fe2O3@C`
- `MoS2/rGO`
- `Ti3C2Tx MXene`

示例：
- `Na3V2(PO4)3/C composite` → MATERIAL: `Na3V2(PO4)3/C`
- `SnS2/graphene composite` → MATERIAL: `SnS2/graphene`

#### （3）掺杂表达中的主体材料
- `Fe-doped Na3V2(PO4)3` → MATERIAL: `Na3V2(PO4)3`

---

### 3.3 不应标注为 MATERIAL 的部分

#### （1）体系描述词
- `composite`
- `based`
- `supported on ...`
- `derived from ...`

#### （2）非活性材料
- 粘结剂：`PVDF`
- 电解液：`NaPF6`
- 集流体：`Al foil`, `Cu foil`
- 添加剂、溶剂等

#### （3）泛指或无明确组成的词
- `material`
- `electrode`
- `sample`
- `product`

#### （4）合成前驱体/反应物（新增项）
- 仅作为反应物出现的物质（如合成过程中消耗的金属单质、还原剂、溶剂）不标为 MATERIAL。
- 示例：`Co can react with Sb2S3 to form CoSbS` → 仅标 `Sb2S3` 和 `CoSbS`，不标前驱体 `Co`。

---

### 3.4 边界规则

- MATERIAL 只包含材料名称本身，不包含：
  - `material`
  - `electrode`
  - `particles`
  - `powder`

示例：
- `Na3V2(PO4)3 particles` → MATERIAL: `Na3V2(PO4)3`
但对于 carbon nanotubes 等公认材料名称，整体标注为 MATERIAL。

**特别注意：**
1. 如果材料名称与**形貌词**结合，形成了紧凑的缩写，则视为专有材料名称，整体标注为 MATERIAL。
例如：`MVO-NBs`, `MVO-NPs`。
2. **（修改项）严禁将掺杂元素前缀与母体合并标注。** 即使出现 `B-NVP/C` 或 `Cl-NVP/C`，也必须将前缀剥离（前缀标为 O），仅将核心母体 `NVP/C` 标为 MATERIAL。

---

## 4. MODIFICATION（化学修饰）

### 4.1 定义

表示对材料进行的**化学层面改性方式**，  
回答“**通过何种化学手段改变材料**”，而非材料本体或晶体结构。

---

### 4.2 应标注为 MODIFICATION

- 掺杂：`doped`, `co-doped`, `Fe-doped`, `Mg co-doped`
- 缺陷：`vacancy`, `oxygen-vacancy`, `defect-rich`
- 取代：`substituted`
- 包覆：`coated`, `carbon-coated`,`coating`
- 价态/化学状态：`Fe2+`, `Mn3+ enriched`, `Ti3+`
- 表面修饰：`surface-modified`, `surface fluorination`

**补充规则（新增项）：**
- **动作绑定原则**：在被动语态中，作为谓语动词（如 `doped`, `coated`）主语的元素或离子必须标为 MODIFICATION。
- 示例：`V-ions are successfully doped` → `V-ions` 标为 MODIFICATION，`doped` 标为 MODIFICATION。

---

### 4.3 不应标注为 MODIFICATION

- 性能形容词：`high-performance`, `excellent`
- 形貌词：`porous`, `nanoparticles`
- 体系描述词：`composite`, `based`, `supported`
- **描述电化学过程、动力学行为或被抑制/促进的现象中出现的离子、空位或价态符号，不作为 MODIFICATION 标注**

说明：  
当相关词语用于描述**充放电过程中发生或被抑制的物理/电化学现象**（如 *vacancy ordering*、*ion migration*、*phase transition*）时，  
即使包含 `vacancy`、`Na+`、`Fe3+` 等形式，也不视为材料的化学修饰属性。

示例对比：
- `oxygen-vacancy rich carbon` → MODIFICATION  
- `suppresses Na+/vacancy ordering` → O 

---

### 4.4 与 STRUCTURE 的裁决规则（重要）

当同一描述同时涉及**化学缺陷/组成变化**与**结构状态**时，遵循以下规则：

> **若该词主要回答“通过何种化学方式引入变化”，标注为 MODIFICATION；  
> 若主要回答“材料是否有序、是否结晶、排列状态如何”，标注为 STRUCTURE。**

示例：
- `oxygen-vacancy carbon`  
  → `oxygen-vacancy` → MODIFICATION  
- `defect-rich disordered carbon`  
  → `defect-rich` → MODIFICATION  
  → `disordered` → STRUCTURE  

---

### 4.5 松散结构的拆分规则（新增项）

**判定准则：介词触发制。** 当改性描述中出现介词（of, with, by, through）导致结构松散时，必须执行**外科手术式拆分**，严禁整体标注。

- **逻辑**：以介词为界，剥离介词，仅提取核心动作与独立的改性元素。
- **示例（松散型 - 拆分）**：`doping of nitrogen and sulfur` 
  → `doping` → MODIFICATION 
  → `nitrogen` → MODIFICATION 
  → `sulfur` → MODIFICATION 
- **对比（紧凑型 - 不拆）**：`nitrogen and sulfur co-doping`
  → 没有任何介词触发，视为整体技术名称，整体标注为 MODIFICATION。

---

## 5. ROLE（电池角色）

### 定义

材料在钠离子电池中的**功能角色**。

### 应标注
- `anode`
- `cathode`
- `negative electrode`
- `positive electrode`

> 当出现 `anode material`、`cathode electrode` 等短语时，  
> **仅标注 `anode` / `cathode` 本身。**

---

## 6. STRUCTURE（晶体 / 相结构）

### 6.1 定义

材料的**晶体结构或相结构类型**，  
回答“**材料以何种结构/相存在**”，而非形貌或修饰方式。

---

### 6.2 应标注为 STRUCTURE

- `layered`
- `spinel`
- `olivine`
- `NASICON`
- `tunnel`
- `Prussian blue framework`
- `disordered`（表示无序/非晶结构状态）
- **晶体空间群符号（新增项）**：如 `P6(3)/mmc`, `Fd-3m`, `R-3m`。

示例：  
layered Na0.67MnO2  
→ `layered` → STRUCTURE  
→ `Na0.67MnO2` → MATERIAL  

---

### 6.3 专有 vs 泛化结构的硬规则（非常重要）

> **只有当某一结构术语可以在材料学文献中作为“独立的晶体结构/相类别”单独出现时，才标注为 STRUCTURE；  
> 否则一律视为泛化描述，标注为 O。**

#### 标注为 STRUCTURE：
- `Prussian blue framework`


#### 统一标注为 O：
- `carbon framework`
- `3D framework`
- `open framework structure`
- `porous framework`

---

### 6.3.1 framework 相关专项说明（补充规则）

当结构描述中包含 **framework** 时，遵循以下补充规则以避免过度泛化：

> **仅当 “X framework” 在材料科学文献中被公认为特定晶体结构或相结构的名称，  
> 且可在不依附具体材料名称的情况下独立使用时，才标注为 STRUCTURE；  
> 否则一律视为泛化结构描述，标注为 O。**

说明：
- 该规则为 **判据性规则**，而非穷举白名单；
- 未在示例中列出的 framework 表达，需按上述判据判断，而非默认标注。

示例：
- `Prussian blue framework` → STRUCTURE  
- `carbon framework` → O  
- `conductive framework` → O  
- `3D framework` → O  

---

### 6.4 明确不标（统一为 O）

以下形貌词**不作为 STRUCTURE 标注**：
- `nanoparticles`
- `nanorods`
- `nanosheets`
- `porous`
- `hollow`
- `microspheres`
- `hierarchical`

---

## 7. 综合示例

### 示例 1
Fe and Mg co-doped Na3V2(PO4)3/C composite as a cathode for sodium-ion batteries.

- `Fe and Mg co-doped` → MODIFICATION 
- `Na3V2(PO4)3/C` → MATERIAL  
- `cathode` → ROLE  

### 示例 2
Layered Na0.67MnO2 as a high-performance cathode.

- `layered` → STRUCTURE  
- `Na0.67MnO2` → MATERIAL  
- `cathode` → ROLE  

### 示例 3
Porous hard carbon anode for Na-ion batteries.

- `hard carbon` → MATERIAL  
- `anode` → ROLE  
（`porous` → O）

### 示例 4
Na2FeMn(CN)6 with a Prussian blue framework as a cathode.

- `Na2FeMn(CN)6` → MATERIAL  
- `Prussian blue framework` → STRUCTURE  
- `cathode` → ROLE  

---

## 8. 总体标注原则

1. **核心优先**：MATERIAL 仅表示活性材料及其组成形式。  
2. **体系不进实体**：`composite`, `based`, `supported` 等不进入 MATERIAL。  
3. **组成可进实体**：`X/C`, `X/graphene`, `X@C` 视为 MATERIAL。  
4. **结构与形貌严格区分**：晶体/相结构 → STRUCTURE，形貌一律 → O。  
5. **修饰不吞主体**：MODIFICATION 不包含 MATERIAL。 
6. **冲突有裁决**：MODIFICATION vs STRUCTURE 按“回答的问题类型”裁定。  
7. **一致性优先**：相同表达在不同上下文中保持统一标注。  
8. **面向自动化**：保证人工标注与 LLM 标注可高度一致。