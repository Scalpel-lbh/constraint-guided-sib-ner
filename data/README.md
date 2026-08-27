# Data documentation

## Public files

`source_articles.csv` contains bibliographic metadata and persistent identifiers for the 2,681 documents used in the study. It does not contain abstract text.

`annotation/` contains the authors' manually created public entity annotations,
separated exactly according to the reported 30/20/50 split:

- `train_30.jsonl`: 30 training documents
- `val_20.jsonl`: 20 validation documents
- `test_50.jsonl`: 50 test documents

These public JSONL files contain document identifiers, abstract hashes, and
entity spans, but no abstract text. The local runtime inputs use the same
directory and filenames ending in `.json`; those files are intentionally absent
until `scripts/prepare_data.py` joins lawfully obtained abstracts and creates
them at the exact paths expected by the code.

`Na/remain_2581_metadata.csv` contains the public identifiers, original ordering,
and abstract hashes for the 2,581-document unlabeled pool. It contains no abstract
text. `prepare_data.py` reconstructs the local `Na/remain_2581.json` runtime
file after the supplied abstracts pass hash verification.

`silver_annotations/` is organized first by method and then by iteration round.
Each method directory contains `round_1.jsonl` through `round_5.jsonl`, with 50
documents in each file and 250 documents per method. The four method directories
are `llm_annotation`, `constraint_guided`, `without_refinement`, and
`without_recovery`. `entities` records the method output; `aligned_entities`
records the exact spans used for downstream training. Context sentences,
complete LLM responses, and abstract text are excluded.

`prepare_data.py` reconstructs each local `LLM/sampled/round_N.json` directly
from the corresponding released `silver_annotations/llm_annotation/round_N.jsonl`
file after the supplied abstracts pass hash verification.

## Entity labels

- `MATERIAL`: active material or material matrix.
- `STRUCTURE`: crystal structure, phase, or structural motif.
- `MODIFICATION`: doping, coating, treatment, or other material modification.
- `ROLE`: functional battery role, such as anode or cathode.

## Document linkage

Each annotation record is linked to `source_articles.csv` through `document_id`.
The `doi` column contains the verified release value. DOI links can be formed as
`https://doi.org/{doi}`. The `abstract_sha256` values stored in the public gold
annotation files and `Na/remain_2581_metadata.csv` are SHA-256 digests of the
normalized abstract text used in the experiments. They enable version
verification without redistributing the text.

The corpus contains three pairs of duplicate bibliographic records representing early-access/final-indexing variants with the same DOI and different Web of Science accession identifiers. They are retained because this repository records the corpus exactly as used in the reported experiments; no post hoc sample removal was performed.

## Restricted material

The original abstracts are not part of this repository. Access to the linked articles does not imply permission to redistribute their text. Users are responsible for obtaining the text through a lawful source and complying with the applicable terms.
