import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


ROOT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = ROOT_DIR / "baseline_65"
LLM_SR_R4_DIR = ROOT_DIR / "LLM_SR" / "iterations" / "round_4"
EXPERIMENT_DIR = ROOT_DIR / "seed_experiments" / "constraint_guided_r4"
SEEDS_TO_RUN = [42, 3407, 2024, 2025, 2026]

sys.path.insert(0, str(BASELINE_DIR))
sys.path.insert(0, str(ROOT_DIR / "LLM_SR"))

from config import TRAINING_PARAM_OVERRIDES  # noqa: E402

sys.modules.pop("config", None)
sys.path = [str(BASELINE_DIR)] + [path for path in sys.path if str(ROOT_DIR / "LLM_SR") not in path]

from config import (  # noqa: E402
    MODEL_NAME,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    LEARNING_RATE as BASE_LEARNING_RATE,
    CRF_LEARNING_RATE as BASE_CRF_LEARNING_RATE,
    NUM_EPOCHS as BASE_NUM_EPOCHS,
    WARMUP_RATIO as BASE_WARMUP_RATIO,
    WEIGHT_DECAY as BASE_WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE as BASE_EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA as BASE_EARLY_STOPPING_MIN_DELTA,
    DROPOUT_RATE as BASE_DROPOUT_RATE,
)
from data_processor import create_dataloaders  # noqa: E402
from model import MatSciBERTCRF  # noqa: E402
from train import EarlyStopping, train_epoch, validate, set_seed  # noqa: E402


TRAIN_FILE = LLM_SR_R4_DIR / "train_merged.json"
VAL_FILE = ROOT_DIR / "data" / "annotation" / "val_20.json"
TEST_FILE = ROOT_DIR / "data" / "annotation" / "test_50.json"

LEARNING_RATE = TRAINING_PARAM_OVERRIDES.get("LEARNING_RATE", BASE_LEARNING_RATE)
CRF_LEARNING_RATE = TRAINING_PARAM_OVERRIDES.get("CRF_LEARNING_RATE", BASE_CRF_LEARNING_RATE)
NUM_EPOCHS = TRAINING_PARAM_OVERRIDES.get("NUM_EPOCHS", BASE_NUM_EPOCHS)
WARMUP_RATIO = TRAINING_PARAM_OVERRIDES.get("WARMUP_RATIO", BASE_WARMUP_RATIO)
WEIGHT_DECAY = TRAINING_PARAM_OVERRIDES.get("WEIGHT_DECAY", BASE_WEIGHT_DECAY)
EARLY_STOPPING_PATIENCE = TRAINING_PARAM_OVERRIDES.get(
    "EARLY_STOPPING_PATIENCE", BASE_EARLY_STOPPING_PATIENCE
)
EARLY_STOPPING_MIN_DELTA = TRAINING_PARAM_OVERRIDES.get(
    "EARLY_STOPPING_MIN_DELTA", BASE_EARLY_STOPPING_MIN_DELTA
)
DROPOUT_RATE = TRAINING_PARAM_OVERRIDES.get("DROPOUT_RATE", BASE_DROPOUT_RATE)


def set_strict_seed(seed: int) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def to_jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_seed(seed: int) -> None:
    output_dir = EXPERIMENT_DIR / f"seed_{seed}"
    result_path = output_dir / "results.json"
    if result_path.exists():
        print(f"Skip seed {seed}: {result_path} already exists")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    set_strict_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Running Constraint-Guided R4 seed {seed} on {device} ===")
    print(f"Train file: {TRAIN_FILE}")
    print(f"Tokenizer/model: {MODEL_NAME}")
    print(
        f"LR={LEARNING_RATE}, dropout={DROPOUT_RATE}, warmup={WARMUP_RATIO}, "
        f"epochs={NUM_EPOCHS}, patience={EARLY_STOPPING_PATIENCE}"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(
        str(TRAIN_FILE),
        str(VAL_FILE),
        str(TEST_FILE),
        tokenizer,
        TRAIN_BATCH_SIZE,
        EVAL_BATCH_SIZE,
    )

    model = MatSciBERTCRF(dropout_rate=DROPOUT_RATE).to(device)

    bert_params = []
    crf_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "crf" in name:
            crf_params.append(param)
        elif "bert" in name:
            bert_params.append(param)
        else:
            other_params.append(param)

    if crf_params:
        optimizer_groups = [
            {"params": bert_params, "lr": LEARNING_RATE},
            {"params": crf_params, "lr": CRF_LEARNING_RATE},
            {"params": other_params, "lr": LEARNING_RATE},
        ]
    else:
        optimizer_groups = [
            {"params": bert_params, "lr": LEARNING_RATE},
            {"params": other_params, "lr": LEARNING_RATE},
        ]

    optimizer = AdamW(optimizer_groups, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        mode="max",
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
    }
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_metrics = validate(model, val_loader, device, val_dataset, desc=f"Epoch {epoch} [Val]")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["micro_f1"])
        history["val_precision"].append(val_metrics["micro_precision"])
        history["val_recall"].append(val_metrics["micro_recall"])

        print(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
            f"Val F1={val_metrics['micro_f1']:.4f}"
        )

        if early_stopping(val_metrics["micro_f1"], epoch):
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_f1": val_metrics["micro_f1"],
                    "history": history,
                },
                best_model_path,
            )

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}; best epoch {early_stopping.best_epoch}")
            break

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = validate(model, test_loader, device, test_dataset, desc="Test")

    results = {
        "round": 4,
        "method": "Constraint-Guided",
        "training_info": {
            "total_train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "best_epoch": early_stopping.best_epoch,
            "best_val_f1": early_stopping.best_score,
        },
        "test_metrics": test_metrics,
        "history": history,
        "config": {
            "model_name": MODEL_NAME,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "crf_learning_rate": CRF_LEARNING_RATE,
            "dropout_rate": DROPOUT_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "num_epochs": NUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "seed": seed,
            "train_file": str(TRAIN_FILE),
        },
    }
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False, default=to_jsonable)

    print(
        f"Saved seed {seed}: F1={test_metrics['micro_f1']:.4f}, "
        f"P={test_metrics['micro_precision']:.4f}, R={test_metrics['micro_recall']:.4f}"
    )


def get_seed_from_result(result: dict, result_path: Path) -> int:
    if "config" in result and "seed" in result["config"]:
        return int(result["config"]["seed"])
    return int(result_path.parent.name.split("_")[1])


def summarize() -> None:
    rows = []
    for result_path in sorted(
        EXPERIMENT_DIR.glob("seed_*/results.json"),
        key=lambda path: int(path.parent.name.split("_")[1]),
    ):
        with result_path.open("r", encoding="utf-8") as file:
            result = json.load(file)
        metrics = result["test_metrics"]
        training_info = result.get("training_info", {})
        rows.append(
            {
                "seed": get_seed_from_result(result, result_path),
                "precision": metrics["micro_precision"],
                "recall": metrics["micro_recall"],
                "f1": metrics["micro_f1"],
                "best_epoch": training_info.get("best_epoch"),
                "best_val_f1": training_info.get("best_val_f1"),
            }
        )

    if not rows:
        return

    summary_path = EXPERIMENT_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    for row in rows:
        print(
            f"seed={row['seed']} "
            f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f} "
            f"best_epoch={row['best_epoch']}"
        )
    print(f"Saved summary: {summary_path}")


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS_TO_RUN:
        run_seed(seed)
    summarize()


if __name__ == "__main__":
    main()
