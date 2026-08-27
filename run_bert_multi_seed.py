import json
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = ROOT_DIR / "baseline_65"
EXPERIMENT_DIR = ROOT_DIR / "seed_experiments" / "bert"
SEEDS_TO_RUN = [42, 3407, 2024, 2025, 2026]


def run_seed(seed: int) -> None:
    output_dir = EXPERIMENT_DIR / f"seed_{seed}"
    result_path = output_dir / "results.json"
    if result_path.exists():
        print(f"Skip seed {seed}: {result_path} already exists")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["BERT_SEED"] = str(seed)
    env["BERT_OUTPUT_DIR"] = str(output_dir)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    print(f"\n=== Running BERT seed {seed} ===")
    subprocess.run(
        [sys.executable, "train.py"],
        cwd=BASELINE_DIR,
        env=env,
        check=True,
    )


def summarize() -> None:
    rows = []
    for result_path in sorted(
        EXPERIMENT_DIR.glob("seed_*/results.json"),
        key=lambda path: int(path.parent.name.split("_")[1]),
    ):
        with result_path.open("r", encoding="utf-8") as file:
            result = json.load(file)
        metrics = result["test_metrics"]
        rows.append(
            {
                "seed": result["config"]["seed"],
                "precision": metrics["micro_precision"],
                "recall": metrics["micro_recall"],
                "f1": metrics["micro_f1"],
                "best_epoch": result["best_epoch"],
                "best_val_f1": result["best_val_f1"],
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
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS_TO_RUN:
        run_seed(seed)
    summarize()


if __name__ == "__main__":
    main()
