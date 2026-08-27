"""Reconstruct local training inputs from legally obtained abstract text.

The public repository intentionally contains no abstract text. Supply a JSON,
JSONL, or CSV file with `document_id` or `doi`, plus an `abstract` field. This
script verifies hashes and creates the paths expected by the original code.
Generated text-bearing files are ignored by Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHOD_DIRS = {
    "llm_annotation": "LLM",
    "constraint_guided": "LLM_SR",
    "without_refinement": "LLM_SR_wo_error_refinement",
    "without_recovery": "LLM_SR_wo_missing_recovery",
}


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value or "")).strip()


def digest(value: str) -> str:
    return hashlib.sha256(normalized_text(value).encode("utf-8")).hexdigest()


def normalized_doi(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    return value


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def load_abstracts(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    if suffix == ".jsonl":
        return load_jsonl(path)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list):
        raise ValueError("Abstract input JSON must contain a list of objects")
    return value


def load_metadata() -> list[dict]:
    with (ROOT / "data" / "source_articles.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        metadata = list(csv.DictReader(stream))

    hashes: dict[str, str] = {}
    for filename in ("train_30.jsonl", "val_20.jsonl", "test_50.jsonl"):
        for row in load_jsonl(ROOT / "data" / "annotation" / filename):
            hashes[row["document_id"]] = row["abstract_sha256"]
    with (ROOT / "data" / "Na" / "remain_2581_metadata.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            hashes[row["document_id"]] = row["abstract_sha256"]

    for row in metadata:
        document_id = row["document_id"]
        if document_id not in hashes:
            raise ValueError(f"Missing abstract hash for {document_id}")
        row["abstract_sha256"] = hashes[document_id]
    return metadata


def index_abstracts(rows: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    by_id, by_doi = {}, {}
    for row in rows:
        abstract = row.get("abstract", "")
        if not abstract:
            continue
        document_id = (row.get("document_id") or "").strip()
        doi = normalized_doi(row.get("doi", ""))
        if document_id:
            by_id[document_id] = abstract
        if doi:
            by_doi[doi] = abstract
    return by_id, by_doi


def get_abstract(meta: dict, by_id: dict[str, str], by_doi: dict[str, str]) -> str | None:
    return (
        by_id.get(meta["document_id"])
        or by_doi.get(normalized_doi(meta.get("doi", "")))
    )


def year_value(meta: dict):
    value = str(meta.get("year", ""))
    return int(value) if value.isdigit() else value


def labelstudio_record(meta: dict, abstract: str, entities: list[dict], year_override=None) -> dict:
    results = []
    for index, entity in enumerate(entities):
        results.append(
            {
                "id": f"public-{index}",
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "manual",
                "value": {
                    "start": entity["start"],
                    "end": entity["end"],
                    "text": entity["text"],
                    "labels": [entity["label"]],
                },
            }
        )
    return {
        "id": meta["document_id"],
        "data": {
            "text": abstract,
            "title": meta["title"],
            "year": year_value(meta) if year_override is None else year_override,
        },
        "annotations": [{"result": results}],
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abstracts", type=Path, required=True, help="Licensed JSON, JSONL, or CSV abstract file")
    parser.add_argument("--allow-hash-mismatch", action="store_true")
    args = parser.parse_args()

    metadata = load_metadata()
    metadata_by_id = {row["document_id"]: row for row in metadata}
    by_id, by_doi = index_abstracts(load_abstracts(args.abstracts))
    resolved: dict[str, str] = {}
    missing, mismatched = [], []
    for meta in metadata:
        abstract = get_abstract(meta, by_id, by_doi)
        if abstract is None:
            missing.append(meta["document_id"])
            continue
        if digest(abstract) != meta["abstract_sha256"]:
            mismatched.append(meta["document_id"])
        resolved[meta["document_id"]] = abstract

    if missing:
        raise SystemExit(f"Missing abstracts for {len(missing)} documents; first IDs: {missing[:10]}")
    if mismatched and not args.allow_hash_mismatch:
        raise SystemExit(
            f"Hash mismatch for {len(mismatched)} documents; exact reproduction requires the same text. "
            f"First IDs: {mismatched[:10]}. Use --allow-hash-mismatch only for non-exact reruns."
        )

    gold = []
    for filename in ("train_30.jsonl", "val_20.jsonl", "test_50.jsonl"):
        gold.extend(load_jsonl(ROOT / "data" / "annotation" / filename))
    gold_by_split: dict[str, list[dict]] = defaultdict(list)
    for row in gold:
        meta = metadata_by_id[row["document_id"]]
        gold_by_split[row["split"]].append(labelstudio_record(meta, resolved[row["document_id"]], row["entities"]))
    split_paths = {"train": "train_30.json", "validation": "val_20.json", "test": "test_50.json"}
    for split, filename in split_paths.items():
        write_json(ROOT / "data" / "annotation" / filename, gold_by_split[split])

    pool = []
    for meta in sorted(
        (x for x in metadata if x["subset"] == "unlabeled_pool"),
        key=lambda x: int(x["unlabeled_pool_index"]),
    ):
        pool.append({"title": meta["title"], "abstract": resolved[meta["document_id"]], "year": year_value(meta)})
    write_json(ROOT / "data" / "Na" / "remain_2581.json", pool)

    for method, directory in METHOD_DIRS.items():
        by_round: dict[int, list[dict]] = defaultdict(list)
        for round_id in range(1, 6):
            rows = load_jsonl(
                ROOT / "data" / "silver_annotations" / method / f"round_{round_id}.jsonl"
            )
            by_round[round_id].extend(rows)
        cumulative = list(gold_by_split["train"])
        for round_id in range(1, 6):
            for row in by_round[round_id]:
                meta = metadata_by_id[row["document_id"]]
                cumulative.append(
                    labelstudio_record(
                        meta,
                        resolved[row["document_id"]],
                        row["aligned_entities"],
                        year_override=row.get("year", ""),
                    )
                )
            write_json(ROOT / directory / "iterations" / f"round_{round_id}" / "train_merged.json", cumulative)

    for round_id in range(1, 6):
        llm_rows = load_jsonl(
            ROOT / "data" / "silver_annotations" / "llm_annotation" / f"round_{round_id}.jsonl"
        )
        samples = []
        for row in (x for x in llm_rows if int(x["round"]) == round_id):
            meta = metadata_by_id[row["document_id"]]
            samples.append(
                {
                    "title": meta["title"],
                    "abstract": resolved[row["document_id"]],
                    "year": year_value(meta),
                    "_sampling_info": row.get("sampling", {}),
                }
            )
        sampled_round = {
            "round": round_id,
            "samples": samples,
        }
        write_json(ROOT / "LLM" / "sampled" / f"round_{round_id}.json", sampled_round)

    print(f"Prepared {len(metadata)} documents; hash mismatches allowed: {len(mismatched)}")


if __name__ == "__main__":
    main()
