"""
evaluate_models.py
------------------
Rigorous evaluation of all three approaches on the held-out test set.

Reports:
  - Accuracy, Precision, Recall, Macro-F1, per-class metrics, confusion matrix.
  - Weak-password recall and false-strong rate.
  - DFA policy statistics on the test set.
  - Hybrid vs SLM-only ablation.
  - Model size, training time, latency (mean and p95).
  - Saves results to models/evaluation_report.json and confusion matrices to models/.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dfa_simulator import load_simulator
from hybrid_engine import load_slm, slm_predict, analyse, CLASS_NAMES, STRONG_TIERS

WEAK_TIERS = {"Very Weak", "Weak"}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_test_set():
    path = os.path.join(BASE_DIR, "data", "processed", "test.csv")
    df = pd.read_csv(path)
    return df


def evaluate_baseline(test_df):
    model_path = os.path.join(BASE_DIR, "models", "baseline_model.joblib")
    if not os.path.exists(model_path):
        return None
    pipeline = joblib.load(model_path)
    preds = pipeline.predict(test_df["password"].astype(str))
    return preds.tolist()


def evaluate_slm_on_test(test_df, model, vocab, device):
    all_preds = []
    latencies = []
    for pw in test_df["password"].astype(str):
        result = slm_predict(pw, model, vocab, device)
        all_preds.append(result["predicted_index"])
        latencies.append(result["latency_ms"])
    return all_preds, latencies


def compute_classification_metrics(y_true, y_pred, name: str):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Weak-password recall: recall on classes 0+1 (Very Weak / Weak)
    weak_mask = np.isin(np.array(y_true), [0, 1])
    weak_preds = np.array(y_pred)[weak_mask]
    weak_true = np.array(y_true)[weak_mask]
    weak_recall = float(np.mean(weak_preds <= 1)) if len(weak_true) > 0 else 0.0

    # False-strong rate: fraction of actually-weak passwords predicted Strong or Very Strong
    false_strong = float(np.mean(weak_preds >= 3)) if len(weak_true) > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  {name} — Classification Metrics")
    print(f"{'='*60}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Macro-Precision:   {prec:.4f}")
    print(f"  Macro-Recall:      {rec:.4f}")
    print(f"  Macro-F1:          {f1:.4f}")
    print(f"  Weak-PW Recall:    {weak_recall:.4f}")
    print(f"  False-Strong Rate: {false_strong:.4f}")
    print(f"\nPer-class Report:\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)}")
    print(f"Confusion Matrix (rows=true, cols=pred):\n{np.array(cm)}")

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(rec, 4),
        "macro_f1": round(f1, 4),
        "weak_pw_recall": round(weak_recall, 4),
        "false_strong_rate": round(false_strong, 4),
        "per_class_report": report,
        "confusion_matrix": cm,
    }


def evaluate_dfa_on_test(test_df, simulator):
    accepted, rejected = 0, 0
    for pw in test_df["password"].astype(str):
        result = simulator.run(pw)
        if result["accepted"]:
            accepted += 1
        else:
            rejected += 1
    total = len(test_df)
    print(f"\n{'='*60}")
    print("  DFA Policy Statistics (on held-out test set)")
    print(f"{'='*60}")
    print(f"  Total:    {total}")
    print(f"  Accepted: {accepted} ({100*accepted/total:.1f}%)")
    print(f"  Rejected: {rejected} ({100*rejected/total:.1f}%)")
    return {"total": total, "accepted": accepted, "rejected": rejected,
            "accept_rate": round(accepted / total, 4)}


def run_latency_benchmark(passwords, model, vocab, device, n_runs=3):
    """Runs inference n_runs times on each password to get stable latency estimates."""
    all_lat = []
    for pw in passwords[:200]:  # Sample 200 for speed
        for _ in range(n_runs):
            r = slm_predict(pw, model, vocab, device)
            all_lat.append(r["latency_ms"])
    return {
        "mean_ms": round(float(np.mean(all_lat)), 3),
        "p50_ms": round(float(np.percentile(all_lat, 50)), 3),
        "p95_ms": round(float(np.percentile(all_lat, 95)), 3),
        "p99_ms": round(float(np.percentile(all_lat, 99)), 3),
    }


def model_size_bytes(path: str):
    return os.path.getsize(path) if os.path.exists(path) else 0


def main():
    print("Loading artefacts...")
    test_df = load_test_set()
    y_true = test_df["label"].astype(int).tolist()

    model_dir = os.path.join(BASE_DIR, "models")
    jff_path = os.path.join(BASE_DIR, "automata", "password_dfa.jff")

    model, vocab, cfg, device = load_slm(model_dir)
    simulator = load_simulator(jff_path)

    report = {}

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n[1/5] Evaluating baseline model...")
    baseline_preds = evaluate_baseline(test_df)
    if baseline_preds is not None:
        report["baseline"] = compute_classification_metrics(y_true, baseline_preds, "Baseline (TF-IDF + LR)")
        baseline_size = model_size_bytes(os.path.join(model_dir, "baseline_model.joblib"))
        report["baseline"]["model_size_bytes"] = baseline_size
    else:
        print("  Baseline model not found, skipping.")

    # ── SLM ───────────────────────────────────────────────────────────────────
    print("\n[2/5] Evaluating SLM on test set...")
    slm_preds, latencies = evaluate_slm_on_test(test_df, model, vocab, device)
    report["slm"] = compute_classification_metrics(y_true, slm_preds, "SLM (Char Transformer)")
    report["slm"]["model_size_bytes"] = model_size_bytes(os.path.join(model_dir, "slm_best.pt"))
    report["slm"]["num_parameters"] = cfg.get("num_params", "N/A")
    report["slm"]["training_time_seconds"] = cfg.get("training_time_seconds", "N/A")
    report["slm"]["latency"] = {
        "mean_ms": round(float(np.mean(latencies)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies, 99)), 3),
    }

    # ── DFA ───────────────────────────────────────────────────────────────────
    print("\n[3/5] Running DFA on test set...")
    report["dfa"] = evaluate_dfa_on_test(test_df, simulator)

    # ── Hybrid verdict (ablation) ─────────────────────────────────────────────
    print("\n[4/5] Computing Hybrid verdicts (ablation vs SLM-only)...")
    hybrid_verdicts, disagreements = [], 0
    for i, pw in enumerate(test_df["password"].astype(str)):
        dfa_r = simulator.run(pw)
        slm_tier = CLASS_NAMES[slm_preds[i]]
        from hybrid_engine import hybrid_verdict
        verdict = hybrid_verdict(dfa_r["accepted"], slm_tier)
        hybrid_verdicts.append(verdict)
        if dfa_r["accepted"] != (slm_preds[i] >= 2):  # DFA accepted but SLM thinks weak
            disagreements += 1

    # Hybrid classification: map verdict back to a class index for comparison
    VERDICT_TO_LABEL = {
        "REJECT_POLICY": 0,
        "REJECT_VULNERABILITY": 0,
        "ACCEPT_CONDITIONAL": 2,
        "ACCEPT_ROBUST": 3,
    }
    hybrid_class_preds = [VERDICT_TO_LABEL[v] for v in hybrid_verdicts]
    report["hybrid_ablation"] = {
        "disagreement_count": disagreements,
        "disagreement_rate": round(disagreements / len(test_df), 4),
        "verdict_distribution": {
            v: hybrid_verdicts.count(v) for v in
            ["REJECT_POLICY", "REJECT_VULNERABILITY", "ACCEPT_CONDITIONAL", "ACCEPT_ROBUST"]
        },
    }
    print(f"  Disagreements between SLM and DFA: {disagreements}/{len(test_df)} "
          f"({100*disagreements/len(test_df):.1f}%)")
    print(f"  Hybrid Verdict Distribution: {report['hybrid_ablation']['verdict_distribution']}")

    # ── Latency benchmark ─────────────────────────────────────────────────────
    print("\n[5/5] Latency benchmark (200 passwords × 3 runs)...")
    report["slm"]["latency_benchmark"] = run_latency_benchmark(
        test_df["password"].tolist(), model, vocab, device
    )
    print(f"  Mean latency: {report['slm']['latency_benchmark']['mean_ms']} ms")
    print(f"  P95 latency:  {report['slm']['latency_benchmark']['p95_ms']} ms")

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = os.path.join(model_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nEvaluation report saved to {report_path}")
    return report


if __name__ == "__main__":
    main()
