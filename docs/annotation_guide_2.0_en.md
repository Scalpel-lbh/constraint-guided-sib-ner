# Sodium-Ion Battery Electrode-Material NER Annotation Guidelines (Revision 2.0)

This document is an English translation of `annotation_guide_2.0.md`, the original Chinese annotation guide used in the study. The translation preserves the entity definitions, boundary rules, exclusion rules, and examples of the original guide.

## 1. Annotation objective

The task is to extract information related to **electrode materials for sodium-ion batteries (SIBs)** from scientific abstracts for training a domain-specific named entity recognition model.

Annotate:

- core active electrode materials and their chemical formulas;
- chemical modification methods applied to materials;
- functional roles in a battery;
- crystal or phase structures of materials.

## 2. Entity types

| Entity type | Definition |
|---|---|
| **MATERIAL** | A core active material or an explicit material composition. |
| **MODIFICATION** | A chemical modification or property-changing treatment applied to a material. |
| **ROLE** | The functional role of a material in a battery. |
| **STRUCTURE** | A crystal structure, phase, or recognized structural type. |

## 3. MATERIAL

### 3.1 Definition

MATERIAL denotes the **active material itself** that plays the principal role in an electrode. It may be:

- a single material;
- an explicit chemical composition;
- an explicit composite or hybrid form, such as `X/C`, `X/graphene`, or `X@C`.

MATERIAL answers the question: **What is the material?**

### 3.2 What to annotate

#### Chemical formulas and established material names

Examples:

- `Na3V2(PO4)3`
- `Na0.67MnO2`
- `NaFePO4`
- `SnS2`
- `hard carbon`
- `graphite`
- `MXene`
- `Prussian blue`

#### Explicit composite or hybrid forms

Annotate the complete material expression:

- `Na3V2(PO4)3/C`
- `SnS2/graphene`
- `Fe2O3@C`
- `MoS2/rGO`
- `Ti3C2Tx MXene`

Boundary examples:

- `Na3V2(PO4)3/C composite` → MATERIAL: `Na3V2(PO4)3/C`
- `SnS2/graphene composite` → MATERIAL: `SnS2/graphene`

#### The material inside a modification expression

- `Fe-doped Na3V2(PO4)3` → MATERIAL: `Na3V2(PO4)3`

### 3.3 What not to annotate as MATERIAL

#### Relational or linking words

Do not include `composite`, `based`, `supported on ...`, or `derived from ...`.

#### Non-active components

Do not annotate binders such as `PVDF`, electrolytes such as `NaPF6`, current collectors such as `Al foil` or `Cu foil`, or additives and solvents that are not the core active material.

#### Generic terms without an explicit composition

Do not annotate generic words such as `material`, `electrode`, `sample`, or `product`.

#### Precursors or reactants before synthesis

Substances that occur only as reactants, intermediates, raw materials, or solvents in a synthesis process are not MATERIAL entities.

- `Co can react with Sb2S3 to form CoSbS` → annotate `Sb2S3` and `CoSbS`, but do not annotate the precursor `Co`.

### 3.4 Boundary rules

Annotate only the material name or formula. Exclude generic trailing words such as `material`, `electrode`, `particles`, and `powder`.

- `Na3V2(PO4)3 particles` → MATERIAL: `Na3V2(PO4)3`

Established material names such as `carbon nanotubes` may be annotated in full.

Special cases:

1. If a compact abbreviation includes a morphology suffix and functions as a specific material name, annotate the full abbreviation, for example `MVO-NBs` or `MVO-NPs`.
2. Do not merge an elemental modification prefix into a composite-material span. For expressions such as `B-NVP/C` or `Cl-NVP/C`, exclude the prefix from MATERIAL and annotate only `NVP/C`; the prefix itself is outside the MATERIAL span.

## 4. MODIFICATION

### 4.1 Definition

MODIFICATION denotes a **chemical treatment or property-changing method** applied to a material. It answers the question: **By what chemical means was the material changed?** It does not describe the material itself or its crystal structure.

### 4.2 What to annotate

- Doping: `doped`, `co-doped`, `Fe-doped`, `Mg co-doped`
- Defects: `vacancy`, `oxygen-vacancy`, `defect-rich`
- Substitution: `substituted`
- Coating: `coated`, `carbon-coated`, `coating`
- Valence or chemical state: `Fe2+`, `Mn3+ enriched`, `Ti3+`
- Other chemical modification: `surface-modified`, `surface fluorination`

#### Independently expressed modifying atoms or ions

When an atom or ion is expressed independently in a passive predicate construction, annotate both the modifier and the modification verb as MODIFICATION.

- `V-ions are successfully doped` → MODIFICATION: `V-ions`; MODIFICATION: `doped`

### 4.3 What not to annotate as MODIFICATION

Do not annotate:

- performance descriptions such as `high-performance` or `excellent`;
- morphology terms such as `porous` or `nanoparticles`;
- relational words such as `composite`, `based`, or `supported`;
- ions, vacancies, valence states, or related expressions when they refer only to an electrochemical process, kinetic behavior, ordering process, or reaction mechanism rather than a chemical modification of the material.

If a term describes a mechanism during charge and discharge—for example, vacancy ordering, ion migration, or phase transition—it is not a material modification even when it contains expressions such as `vacancy`, `Na+`, or `Fe3+`.

Contrast:

- `oxygen-vacancy rich carbon` → `oxygen-vacancy` is MODIFICATION
- `suppresses Na+/vacancy ordering` → no MODIFICATION entity

### 4.4 Deciding between MODIFICATION and STRUCTURE

When an expression contains both a chemical defect or compositional change and a structural state, apply this rule:

> If the term mainly answers how the material was chemically changed, annotate it as MODIFICATION. If it mainly describes whether the material is crystalline, amorphous, or in another structural state, annotate it as STRUCTURE.

Examples:

- `oxygen-vacancy carbon` → MODIFICATION: `oxygen-vacancy`
- `defect-rich disordered carbon` → MODIFICATION: `defect-rich`; STRUCTURE: `disordered`

### 4.5 Discontinuous modification expressions

Use **prepositions as boundaries**. When a modification expression is separated by a preposition such as `of`, `with`, `by`, or `through`, split it into the smallest meaningful spans. Do not merge a discontinuous expression into one entity.

Example—discontinuous expression, split into separate entities:

- `doping of nitrogen and sulfur`
  - MODIFICATION: `doping`
  - MODIFICATION: `nitrogen`
  - MODIFICATION: `sulfur`

Contrast—continuous expression, keep as one entity:

- `nitrogen and sulfur co-doping` → one MODIFICATION span

## 5. ROLE

### 5.1 Definition

ROLE denotes the **functional role of a material in a battery**.

Annotate `anode`, `cathode`, `negative electrode`, and `positive electrode`.

For phrases such as `anode material` or `cathode electrode`, annotate only `anode` or `cathode`.

## 6. STRUCTURE

### 6.1 Definition

STRUCTURE denotes a material's **crystal structure, phase, or recognized structural type**. It answers the question: **What structure or phase does the material have?** It does not describe morphology or chemical modification.

### 6.2 What to annotate

- `layered`
- `spinel`
- `olivine`
- `NASICON`
- `tunnel`
- `Prussian blue framework`
- `disordered`, when it describes an amorphous or non-crystalline structural state
- space-group symbols such as `P6(3)/mmc`, `Fd-3m`, and `R-3m`

- `layered Na0.67MnO2` → STRUCTURE: `layered`; MATERIAL: `Na0.67MnO2`

### 6.3 Domain-specific versus generic structures

Annotate a structural expression only when it is recognized in materials science as a specific crystal structure or phase category. A generic structural description is labeled O.

Annotate as STRUCTURE:

- `Prussian blue framework`

Label as O:

- `carbon framework`
- `3D framework`
- `open framework structure`
- `porous framework`

#### Additional rule for expressions containing “framework”

Annotate `X framework` as STRUCTURE only when it is recognized as the name of a specific crystal structure or structural family and is used with that meaning in the materials-science context. Otherwise, label it O.

This is an open-set rule rather than a closed dictionary. Unlisted framework expressions must be judged from domain knowledge and should not be annotated by default.

Examples:

- `Prussian blue framework` → STRUCTURE
- `carbon framework` → O
- `conductive framework` → O
- `3D framework` → O

### 6.4 Morphology terms are not STRUCTURE

Always label the following morphology terms as O rather than STRUCTURE:

- `nanoparticles`
- `nanorods`
- `nanosheets`
- `porous`
- `hollow`
- `microspheres`
- `hierarchical`

## 7. Combined examples

### Example 1

`Fe and Mg co-doped Na3V2(PO4)3/C composite as a cathode for sodium-ion batteries.`

- MODIFICATION: `Fe and Mg co-doped`
- MATERIAL: `Na3V2(PO4)3/C`
- ROLE: `cathode`

### Example 2

`Layered Na0.67MnO2 as a high-performance cathode.`

- STRUCTURE: `Layered`
- MATERIAL: `Na0.67MnO2`
- ROLE: `cathode`

### Example 3

`Porous hard carbon anode for Na-ion batteries.`

- MATERIAL: `hard carbon`
- ROLE: `anode`
- `Porous` → O

### Example 4

`Na2FeMn(CN)6 with a Prussian blue framework as a cathode.`

- MATERIAL: `Na2FeMn(CN)6`
- STRUCTURE: `Prussian blue framework`
- ROLE: `cathode`

## 8. General annotation principles

1. **Core-first principle:** MATERIAL represents the active material and its explicit composition.
2. **Exclude relational words:** words such as `composite`, `based`, and `supported` are not part of MATERIAL.
3. **Keep explicit composite forms intact:** annotate forms such as `X/C`, `X/graphene`, and `X@C` as MATERIAL.
4. **Distinguish structure from morphology:** crystal or phase structures are STRUCTURE; morphology is O.
5. **Do not merge modification and material:** MODIFICATION and MATERIAL are separate entities.
6. **Resolve conflicts by semantic function:** distinguish MODIFICATION from STRUCTURE according to what the term describes.
7. **Maintain consistency:** annotate the same term consistently when it is used with the same meaning.
8. **Prefer reproducible decisions:** apply the rules consistently across both human and LLM-generated annotations.
