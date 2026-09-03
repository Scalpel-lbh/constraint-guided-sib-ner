# Constraint-Guided Refinement and Recovery for Low-Resource SIB NER

This repository provides the code and annotation data for **“Constraint-Guided Refinement and Recovery of LLM-Generated Annotations for Low-Resource NER in Sodium-Ion Battery Materials Literature.”**

The source abstracts are not redistributed. They were obtained through Web of Science and remain subject to the access, copyright, and licensing conditions of Web of Science and the respective publishers. The repository contains document identifiers, bibliographic metadata, normalized-text hashes, author-generated entity annotations, and the code needed to reconstruct and run the experiments after the user supplies lawfully obtained abstracts.



## Repository contents

- `data/source_articles.csv`: metadata and identifiers for all 2,681 corpus documents; no abstract text.
- `data/annotation/train_30.jsonl`: public entity annotations for the 30-document training split.
- `data/annotation/val_20.jsonl`: public entity annotations for the 20-document validation split.
- `data/annotation/test_50.jsonl`: public entity annotations for the 50-document test split.
- `data/Na/remain_2581_metadata.csv`: identifiers, original ordering, and abstract hashes for the 2,581-document unlabeled pool.
- `data/silver_annotations/<method>/round_1.jsonl` through `round_5.jsonl`: sanitized silver annotations, separated by method and iterative round.
- `docs/annotation_guide_2.0_en.md`: English translation of the entity definitions and annotation rules.
- `docs/annotation_guide_2.0.md`: original Chinese annotation guide used in the study.
- `scripts/prepare_data.py`: verifies supplied abstracts and reconstructs the local text-bearing runtime files.
- `baseline_65/`, `baseline_scibert/`, and `baseline_matscibert/`: supervised NER baselines.
- `baseline_llm_direct/`: direct LLM extraction baseline.
- `LLM/`: targeted sampling and iterative LLM annotation pipeline.
- `LLM_SR/`: complete constraint-guided refinement and missing-entity recovery method.
- `LLM_SR_wo_error_refinement/` and `LLM_SR_wo_missing_recovery/`: ablation variants.
- `error_analysis/error_analysis.py`: error-analysis implementation.

## Environment

The experiments use Python 3.12. Create an isolated environment and install the recorded dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

LLM-based experiments require a DeepSeek API key:

```bash
export DEEPSEEK_API_KEY="your-key"  # Windows PowerShell: $env:DEEPSEEK_API_KEY="your-key"
```

The release uses `deepseek-v4-flash` with thinking explicitly disabled, preserving the non-thinking API behavior used by the original experiment code.

## Reconstructing the local runtime data

Exact reconstruction requires all 2,681 abstracts used in the study. Researchers must obtain these abstracts independently through a source for which they have lawful access.

Prepare a JSON, JSONL, or CSV file containing:

- `document_id`: the Web of Science accession identifier recorded in `data/source_articles.csv`;
- `abstract`: the corresponding abstract text.

Use `document_id` for exact corpus reconstruction. DOI-only matching is not sufficient for the complete corpus because three records have no DOI and three DOI values occur in duplicate bibliographic records.

Run from the repository root:

```bash
python scripts/prepare_data.py --abstracts /path/to/licensed_abstracts.jsonl
```

The script normalizes each supplied abstract, verifies its SHA-256 hash, and stops if a document is missing or its text differs from the version used in the study. It then creates:

- `data/annotation/train_30.json`, `val_20.json`, and `test_50.json`;
- `data/Na/remain_2581.json`;
- `LLM/sampled/round_N.json`, reconstructed from the released per-round silver annotations;
- cumulative `iterations/round_N/train_merged.json` files for all four iterative methods.

These generated files contain licensed abstract text and are excluded by `.gitignore`.

## Reproducing downstream NER results

After data reconstruction, run the supervised baselines:

```bash
python baseline_65/train.py
python baseline_scibert/train.py
python baseline_matscibert/train.py
```

Run the five-seed experiments with seeds 42, 3407, 2024, 2025, and 2026:

```bash
python run_bert_multi_seed.py
python run_scibert_multi_seed.py
python run_matscibert_multi_seed.py
python run_llm_r5_multi_seed.py
python run_llm_sr_r4_multi_seed.py
```

The iterative methods are evaluated at the reporting rounds used in the manuscript: R5 for LLM Annotation and R4 for the complete constraint-guided framework. Generated metrics and checkpoints are written beneath `seed_experiments/` and are not distributed.

## Rerunning LLM annotation and refinement

The public silver annotations allow downstream training to be reconstructed without making new API calls. To regenerate the LLM annotations instead, first reconstruct the local runtime data and then run each round in order:

```bash
python LLM/keyword_sampler.py 1
python LLM/llm_annotator.py 1
python LLM/iteration_train.py 1

python LLM_SR/llm_annotator_sr.py 1
python LLM_SR/iteration_train.py 1
```

Repeat the commands for rounds 1 through 5. The two ablation directories expose the same command interface. Because DeepSeek is a hosted model, newly generated annotations are not guaranteed to be byte-identical across service updates; the released silver annotations and aligned spans record the data actually used for downstream reconstruction.

## Data organization and privacy

Gold annotations retain the original 30/20/50 split. Silver annotations are stored separately for `llm_annotation`, `constraint_guided`, `without_refinement`, and `without_recovery`, with 50 documents per round for five rounds. The `entities` field records the method output, while `aligned_entities` records the exact character spans used for downstream BIO conversion.

Public annotation files contain no abstract text, context snippets, or complete LLM responses. For constraint-guided annotations, `original_entities`, when present, records the entity list before refinement, while `entities` records the refined method output and `aligned_entities` records the spans used for downstream training. `abstract_sha256` links each record to the exact normalized abstract used in the experiments without redistributing that text.
