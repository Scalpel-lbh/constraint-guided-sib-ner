import csv
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "baseline_65"
sys.path.insert(0, str(BASELINE_DIR))

from model import MatSciBERTCRF


TEST_FILE = PROJECT_ROOT / "data" / "annotation" / "test_50.json"
OUTPUT_DIR = PROJECT_ROOT / "error_analysis" / "outputs"

MODELS = {
    "LLM_R5": {
        "path": PROJECT_ROOT / "LLM" / "iterations" / "round_5" / "best_model.pt",
        "dropout_rate": 0.2,
    },
    "Full_R4": {
        "path": PROJECT_ROOT / "LLM_SR" / "iterations" / "round_4" / "best_model.pt",
        "dropout_rate": 0.2,
    },
}

MODEL_NAME = "bert-base-cased"
MAX_LEN = 512
NUM_LABELS = 9
ERROR_TYPES = [
    "Entanglement",
    "Over_span_Boundary",
    "Under_span_Boundary",
    "Type_Error",
    "Missing",
    "Spurious_Entity",
]

GENERIC_INVALID_TERMS = {
    "material",
    "materials",
    "cathode material",
    "cathode materials",
    "anode material",
    "anode materials",
    "electrode material",
    "electrode materials",
    "positive electrode material",
    "positive electrode materials",
    "negative electrode material",
    "negative electrode materials",
    "compound",
    "compounds",
    "composite",
    "composites",
    "sample",
    "samples",
    "precursor",
    "precursors",
}

INVALID_PHRASE_PATTERNS = [
    "electrolyte",
    "separator",
    "binder",
    "current collector",
    "additive",
    "solvent",
    "substrate",
    "film",
    "full cell",
    "battery",
    "batteries",
    "capacity",
    "performance",
    "stability",
    "conductivity",
    "capability",
    "rate capability",
    "cycle life",
    "energy density",
    "power density",
]

MATERIAL_INVALID_PATTERNS = [
    "electrolyte",
    "separator",
    "binder",
    "current collector",
    "electrode",
    "cathode",
    "anode",
]

TRUNCATED_SUFFIXES = (
    "-",
    "/",
    " and",
    " or",
    " with",
    " of",
)

MODIFICATION_CUES = [
    "coated",
    "coating",
    "doped",
    "doping",
    "modified",
    "modification",
    "substituted",
    "substitution",
    "decorated",
    "encapsulated",
    "confined",
    "introduced",
    "grafted",
    "protected",
]

STRUCTURE_CUES = [
    "p2",
    "p3",
    "o2",
    "o3",
    "nasicon",
    "layered",
    "spinel",
    "tunnel",
    "olivine",
    "prussian",
]

ID2LABEL = {
    0: "O",
    1: "B-MATERIAL",
    2: "I-MATERIAL",
    3: "B-STRUCTURE",
    4: "I-STRUCTURE",
    5: "B-MODIFICATION",
    6: "I-MODIFICATION",
    7: "B-ROLE",
    8: "I-ROLE",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_data():
    with open(TEST_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def load_model(checkpoint_path, dropout_rate):
    model = MatSciBERTCRF(
        num_labels=NUM_LABELS,
        model_name=MODEL_NAME,
        dropout_rate=dropout_rate,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def extract_gold_entities(doc):
    annotations = doc.get("annotations", [{}])[0].get("result", [])
    entities = []

    for annotation in annotations:
        value = annotation.get("value")
        if not value or "labels" not in value or not value["labels"]:
            continue

        entities.append(
            {
                "start": value["start"],
                "end": value["end"],
                "text": value["text"],
                "label": value["labels"][0],
            }
        )

    return entities


def extract_entities_from_prediction(text, offset_mapping, predictions):
    entities = []
    current_entity = None

    for pred_id, offset in zip(predictions, offset_mapping):
        if offset[0] == 0 and offset[1] == 0:
            continue

        label = ID2LABEL.get(pred_id, "O")

        if label.startswith("B-"):
            if current_entity:
                current_entity["text"] = text[current_entity["start"] : current_entity["end"]]
                entities.append(current_entity)
            current_entity = {
                "label": label[2:],
                "start": offset[0],
                "end": offset[1],
            }
        elif label.startswith("I-") and current_entity and label[2:] == current_entity["label"]:
            current_entity["end"] = max(current_entity["end"], offset[1])
        else:
            if current_entity:
                current_entity["text"] = text[current_entity["start"] : current_entity["end"]]
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        current_entity["text"] = text[current_entity["start"] : current_entity["end"]]
        entities.append(current_entity)

    return entities


def spans_overlap(entity_a, entity_b):
    return not (entity_a["end"] <= entity_b["start"] or entity_a["start"] >= entity_b["end"])


def overlap_length(entity_a, entity_b):
    return max(0, min(entity_a["end"], entity_b["end"]) - max(entity_a["start"], entity_b["start"]))


def entity_length(entity):
    return max(1, entity["end"] - entity["start"])


def span_iou(entity_a, entity_b):
    intersection = overlap_length(entity_a, entity_b)
    union = max(entity_a["end"], entity_b["end"]) - min(entity_a["start"], entity_b["start"])
    return intersection / max(1, union)


def is_exact_span(entity_a, entity_b):
    return entity_a["start"] == entity_b["start"] and entity_a["end"] == entity_b["end"]


def is_strong_overlap(source, target, threshold=0.5):
    return overlap_length(source, target) / entity_length(target) >= threshold


def normalize_text(text):
    return " ".join(text.lower().replace("-", " ").replace("/", " / ").split())


def has_any_term(text, terms):
    normalized = normalize_text(text)
    return any(term in normalized for term in terms)


def is_invalid_prediction(pred):
    text = normalize_text(pred["text"])
    if text in GENERIC_INVALID_TERMS:
        return True

    if pred["label"] == "MATERIAL" and any(pattern in text for pattern in INVALID_PHRASE_PATTERNS):
        return True

    if pred["label"] == "MATERIAL" and any(pattern in text for pattern in MATERIAL_INVALID_PATTERNS):
        return True

    if pred["label"] in {"MATERIAL", "STRUCTURE", "MODIFICATION"} and len(text) <= 1:
        return True

    if pred["text"].strip().lower().endswith(TRUNCATED_SUFFIXES):
        return True

    return False


def is_entangled_prediction(pred, golds):
    labels = {gold["label"] for gold in golds}

    if "MATERIAL" in labels and ("MODIFICATION" in labels or "STRUCTURE" in labels):
        return True

    return False


def classify_boundary(pred, gold):
    pred_contains_gold = pred["start"] <= gold["start"] and pred["end"] >= gold["end"]
    gold_contains_pred = gold["start"] <= pred["start"] and gold["end"] >= pred["end"]

    if pred_contains_gold and not gold_contains_pred:
        return "Over_span_Boundary"
    if gold_contains_pred and not pred_contains_gold:
        return "Under_span_Boundary"

    pred_extra = entity_length(pred) - overlap_length(pred, gold)
    gold_missing = entity_length(gold) - overlap_length(pred, gold)
    if pred_extra >= gold_missing:
        return "Over_span_Boundary"
    return "Under_span_Boundary"


def analyze_errors(gold_entities, pred_entities):
    errors = {error_type: [] for error_type in ERROR_TYPES}
    assigned_golds = set()
    assigned_preds = set()

    for pred_index, pred in enumerate(pred_entities):
        for gold_index, gold in enumerate(gold_entities):
            if gold_index in assigned_golds:
                continue
            if is_exact_span(pred, gold) and pred["label"] == gold["label"]:
                assigned_preds.add(pred_index)
                assigned_golds.add(gold_index)
                break

    for pred_index, pred in enumerate(pred_entities):
        if pred_index in assigned_preds:
            continue
        for gold_index, gold in enumerate(gold_entities):
            if gold_index in assigned_golds:
                continue
            if is_exact_span(pred, gold):
                errors["Type_Error"].append({"pred": pred, "gold": gold})
                assigned_preds.add(pred_index)
                assigned_golds.add(gold_index)
                break

    for pred_index, pred in enumerate(pred_entities):
        if pred_index in assigned_preds:
            continue

        overlapping_gold_indices = []
        for gold_index, gold in enumerate(gold_entities):
            if gold_index in assigned_golds or not spans_overlap(pred, gold):
                continue

            if is_strong_overlap(pred, gold) or gold["label"] in {"MATERIAL", "MODIFICATION", "STRUCTURE"}:
                overlapping_gold_indices.append(gold_index)

        overlapping_golds = [gold_entities[index] for index in overlapping_gold_indices]
        if overlapping_golds and is_entangled_prediction(pred, overlapping_golds):
            errors["Entanglement"].append(
                {
                    "pred": pred,
                    "golds": overlapping_golds,
                }
            )
            assigned_preds.add(pred_index)
            assigned_golds.update(overlapping_gold_indices)

    candidate_pairs = []
    for pred_index, pred in enumerate(pred_entities):
        if pred_index in assigned_preds:
            continue

        for gold_index, gold in enumerate(gold_entities):
            if gold_index in assigned_golds or not spans_overlap(pred, gold):
                continue

            candidate_pairs.append(
                {
                    "pred_index": pred_index,
                    "gold_index": gold_index,
                    "same_type": pred["label"] == gold["label"],
                    "iou": span_iou(pred, gold),
                    "overlap": overlap_length(pred, gold),
                }
            )

    candidate_pairs.sort(
        key=lambda pair: (
            pair["same_type"],
            pair["iou"],
            pair["overlap"],
        ),
        reverse=True,
    )

    for pair in candidate_pairs:
        pred_index = pair["pred_index"]
        gold_index = pair["gold_index"]

        if pred_index in assigned_preds or gold_index in assigned_golds:
            continue

        pred = pred_entities[pred_index]
        gold = gold_entities[gold_index]

        if pred["label"] == gold["label"]:
            boundary_type = classify_boundary(pred, gold)
            errors[boundary_type].append({"pred": pred, "gold": gold})
        else:
            errors["Type_Error"].append({"pred": pred, "gold": gold})

        assigned_preds.add(pred_index)
        assigned_golds.add(gold_index)

    for gold_index, gold in enumerate(gold_entities):
        if gold_index not in assigned_golds:
            errors["Missing"].append({"gold": gold})

    for pred_index, pred in enumerate(pred_entities):
        if pred_index not in assigned_preds:
            error_record = {"pred": pred}
            error_record["subtype"] = "Invalid_Entity" if is_invalid_prediction(pred) else "Spurious_Entity"
            errors["Spurious_Entity"].append(error_record)

    return errors


def get_last_valid_offset(tokens, offset_mapping, text):
    for index in range(len(offset_mapping) - 1, -1, -1):
        if offset_mapping[index] != [0, 0] or tokens[index] == "[SEP]":
            if tokens[index] == "[SEP]" and index > 0:
                return offset_mapping[index - 1][1]
            return offset_mapping[index][1]
    return len(text)


def predict_entities(model, tokenizer, text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"][0].tolist()

    logits = model(input_ids, attention_mask)
    predictions = torch.argmax(logits, dim=-1)[0].tolist()
    pred_entities = extract_entities_from_prediction(text, offset_mapping, predictions)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    last_valid_offset = get_last_valid_offset(tokens, offset_mapping, text)

    return pred_entities, last_valid_offset


def write_summary(summary_counts):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_json = OUTPUT_DIR / "summary_counts.json"
    summary_csv = OUTPUT_DIR / "summary_counts.csv"
    comparison_csv = OUTPUT_DIR / "comparison_summary.csv"

    with open(summary_json, "w", encoding="utf-8") as file:
        json.dump(summary_counts, file, ensure_ascii=False, indent=2)

    with open(summary_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", *ERROR_TYPES, "total_errors"])
        for model_name, counts in summary_counts.items():
            writer.writerow(
                [
                    model_name,
                    *[counts[error_type] for error_type in ERROR_TYPES],
                    sum(counts.values()),
                ]
            )

    with open(comparison_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["error_type", "LLM_R5", "Full_R4", "reduction", "change_rate"])
        for error_type in ERROR_TYPES:
            llm_count = summary_counts["LLM_R5"][error_type]
            full_count = summary_counts["Full_R4"][error_type]
            reduction = llm_count - full_count
            change_rate = ((full_count - llm_count) / llm_count * 100) if llm_count else 0.0
            writer.writerow([error_type, llm_count, full_count, reduction, f"{change_rate:.2f}%"])

        llm_total = sum(summary_counts["LLM_R5"].values())
        full_total = sum(summary_counts["Full_R4"].values())
        total_reduction = llm_total - full_total
        total_change_rate = ((full_total - llm_total) / llm_total * 100) if llm_total else 0.0
        writer.writerow(["Total", llm_total, full_total, total_reduction, f"{total_change_rate:.2f}%"])


def write_details(details):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    details_file = OUTPUT_DIR / "detailed_errors.json"

    with open(details_file, "w", encoding="utf-8") as file:
        json.dump(details, file, ensure_ascii=False, indent=2)


def main():
    print("Initializing error analysis...")
    print(f"Device: {device}")
    print(f"Test file: {TEST_FILE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_data = load_test_data()

    models = {}
    for model_name, model_config in MODELS.items():
        print(f"Loading {model_name}: {model_config['path']}")
        models[model_name] = load_model(
            model_config["path"],
            dropout_rate=model_config["dropout_rate"],
        )

    summary_counts = {
        model_name: {error_type: 0 for error_type in ERROR_TYPES}
        for model_name in MODELS
    }
    detailed_errors = []

    with torch.no_grad():
        for doc_index, doc in enumerate(test_data):
            text = doc["data"]["text"] if "data" in doc else doc["text"]
            gold_entities = extract_gold_entities(doc)
            doc_record = {
                "doc_index": doc_index,
                "text": text,
                "gold_entities": gold_entities,
                "models": {},
            }

            for model_name, model in models.items():
                pred_entities, last_valid_offset = predict_entities(model, tokenizer, text)
                valid_gold_entities = [
                    gold for gold in gold_entities if gold["start"] < last_valid_offset
                ]
                errors = analyze_errors(valid_gold_entities, pred_entities)

                for error_type in ERROR_TYPES:
                    summary_counts[model_name][error_type] += len(errors[error_type])

                doc_record["models"][model_name] = {
                    "last_valid_offset": last_valid_offset,
                    "pred_entities": pred_entities,
                    "errors": errors,
                }

            detailed_errors.append(doc_record)

    write_summary(summary_counts)
    write_details(detailed_errors)

    print("\n" + "=" * 78)
    print("ERROR ANALYSIS RESULTS (Absolute Counts)")
    print("=" * 78)
    print(f"{'Error Type':<22} | {'LLM_R5':>8} | {'Full_R4':>8} | {'Reduction':>9} | {'Change':>9}")
    print("-" * 78)

    for error_type in ERROR_TYPES:
        llm_count = summary_counts["LLM_R5"][error_type]
        full_count = summary_counts["Full_R4"][error_type]
        reduction = llm_count - full_count
        change_rate = ((full_count - llm_count) / llm_count * 100) if llm_count else 0.0
        print(
            f"{error_type:<22} | {llm_count:>8} | {full_count:>8} | "
            f"{reduction:>+9} | {change_rate:>+8.2f}%"
        )

    print("-" * 78)
    llm_total = sum(summary_counts["LLM_R5"].values())
    full_total = sum(summary_counts["Full_R4"].values())
    total_change_rate = ((full_total - llm_total) / llm_total * 100) if llm_total else 0.0
    print(
        f"{'Total':<22} | {llm_total:>8} | {full_total:>8} | "
        f"{llm_total - full_total:>+9} | {total_change_rate:>+8.2f}%"
    )
    print("=" * 78)
    print(f"Saved summary to: {OUTPUT_DIR / 'summary_counts.csv'}")
    print(f"Saved comparison to: {OUTPUT_DIR / 'comparison_summary.csv'}")
    print(f"Saved details to: {OUTPUT_DIR / 'detailed_errors.json'}")


if __name__ == "__main__":
    main()
