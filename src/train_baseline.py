"""
train_baseline.py
Baseline password-strength classifier: char n-gram TF-IDF + Logistic Regression.
Passwords are treated as sensitive data and never printed or logged.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BaselineConfig:
    ngram_range: tuple = (2, 4)
    max_features: int = 50_000
    C_values: list = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    random_state: int = 42
    class_weight: str = "balanced"
    max_iter: int = 1000


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class CharNgramExtractor:
    """TF-IDF character n-gram extractor.

    Passwords are never stored or logged — only their vectorised representations
    are kept in memory.
    """

    def __init__(self, config: BaselineConfig) -> None:
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=config.ngram_range,
            max_features=config.max_features,
            sublinear_tf=True,
        )

    # ------------------------------------------------------------------
    def fit_transform(self, passwords_train: list) -> sp.csr_matrix:
        """Fit the vectorizer on training passwords and return feature matrix."""
        logger.info(
            "Fitting char-ngram TF-IDF on %d samples …", len(passwords_train)
        )
        return self._vectorizer.fit_transform(passwords_train)

    def transform(self, passwords: list) -> sp.csr_matrix:
        """Transform passwords using the already-fitted vectorizer."""
        return self._vectorizer.transform(passwords)

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._vectorizer, path)
        logger.info("Saved CharNgramExtractor → %s", path)

    @classmethod
    def load(cls, path: Path) -> "CharNgramExtractor":
        path = Path(path)
        obj = cls.__new__(cls)
        obj._vectorizer = joblib.load(path)
        logger.info("Loaded CharNgramExtractor ← %s", path)
        return obj


# ---------------------------------------------------------------------------
# Hand-crafted features
# ---------------------------------------------------------------------------

def _entropy(password: str) -> float:
    """Shannon entropy of the password characters."""
    if not password:
        return 0.0
    counts = np.array([password.count(c) for c in set(password)], dtype=float)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _max_consecutive(password: str) -> int:
    """Length of the longest run of identical consecutive characters."""
    if not password:
        return 0
    max_run, cur_run = 1, 1
    for i in range(1, len(password)):
        if password[i] == password[i - 1]:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 1
    return max_run


def extract_handcrafted_features(passwords: list) -> np.ndarray:
    """Return an (N, 8) float64 array of hand-crafted password features.

    Features (per password):
        0  length
        1  num_upper
        2  num_lower
        3  num_digits
        4  num_special
        5  entropy          (Shannon, base-2)
        6  unique_ratio     (unique chars / length)
        7  max_consecutive  (longest run of identical chars)

    Raw passwords are *never* stored or emitted in the output array.
    """
    rows = []
    for pw in passwords:
        length = len(pw)
        num_upper = sum(1 for c in pw if c.isupper())
        num_lower = sum(1 for c in pw if c.islower())
        num_digits = sum(1 for c in pw if c.isdigit())
        num_special = sum(1 for c in pw if not c.isalnum())
        ent = _entropy(pw)
        unique_ratio = len(set(pw)) / length if length > 0 else 0.0
        max_cons = _max_consecutive(pw)
        rows.append(
            [length, num_upper, num_lower, num_digits, num_special,
             ent, unique_ratio, max_cons]
        )
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class BaselineClassifier:
    """Logistic Regression classifier with cross-validation over C."""

    def __init__(self, config: BaselineConfig) -> None:
        self.config = config
        self._model: Optional[LogisticRegression] = None
        self._best_C: Optional[float] = None

    # ------------------------------------------------------------------
    def train(
        self,
        X_train_feat,
        y_train: np.ndarray,
        X_val_feat,
        y_val: np.ndarray,
    ) -> dict:
        """Train by sweeping C values; keep the model with best val F1-macro.

        Returns a dict with best_C and validation metrics.
        """
        best_f1 = -1.0
        best_model = None
        best_C = self.config.C_values[0]
        sweep_results = []

        for C in self.config.C_values:
            logger.info("Training LR with C=%.4f …", C)
            clf = LogisticRegression(
                C=C,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
                max_iter=self.config.max_iter,
                solver="saga",
                n_jobs=-1,
            )
            clf.fit(X_train_feat, y_train)
            val_preds = clf.predict(X_val_feat)
            f1 = f1_score(y_val, val_preds, average="macro")
            acc = accuracy_score(y_val, val_preds)
            logger.info("  C=%.4f → val F1-macro=%.4f, val acc=%.4f", C, f1, acc)
            sweep_results.append({"C": C, "val_f1_macro": f1, "val_acc": acc})
            if f1 > best_f1:
                best_f1 = f1
                best_model = clf
                best_C = C

        self._model = best_model
        self._best_C = best_C

        val_preds = self._model.predict(X_val_feat)
        metrics = self._compute_metrics(y_val, val_preds)
        metrics["best_C"] = best_C
        metrics["sweep"] = sweep_results
        logger.info("Best C=%.4f → val F1-macro=%.4f", best_C, metrics["f1_macro"])
        return metrics

    # ------------------------------------------------------------------
    def predict(self, X_feat) -> np.ndarray:
        assert self._model is not None, "Model not yet trained."
        return self._model.predict(X_feat)

    def predict_proba(self, X_feat) -> np.ndarray:
        assert self._model is not None, "Model not yet trained."
        return self._model.predict_proba(X_feat)

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "precision_macro": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "per_class": classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self._model, "best_C": self._best_C, "config": self.config},
            path,
        )
        logger.info("Saved BaselineClassifier → %s", path)

    @classmethod
    def load(cls, path: Path) -> "BaselineClassifier":
        path = Path(path)
        payload = joblib.load(path)
        obj = cls(payload["config"])
        obj._model = payload["model"]
        obj._best_C = payload["best_C"]
        logger.info("Loaded BaselineClassifier ← %s", path)
        return obj


# ---------------------------------------------------------------------------
# Combined feature matrix helper
# ---------------------------------------------------------------------------

def _build_feature_matrix(tfidf_feats: sp.csr_matrix, hc_feats: np.ndarray):
    """Horizontally stack sparse TF-IDF features and dense hand-crafted features."""
    hc_sparse = sp.csr_matrix(hc_feats)
    return sp.hstack([tfidf_feats, hc_sparse], format="csr")


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def _load_split(data_dir: Path, split: str):
    """Load passwords (as list of str) and labels (np.ndarray) for a split."""
    pw_path = data_dir / f"pw_{split}.npy"
    y_path = data_dir / f"y_{split}.npy"
    passwords = np.load(pw_path, allow_pickle=True).tolist()
    y = np.load(y_path)
    return passwords, y


def train_baseline(
    data_dir: Path,
    models_dir: Path,
    config: Optional[BaselineConfig] = None,
) -> dict:
    """End-to-end training pipeline for the baseline classifier.

    Args:
        data_dir:   Directory containing pw_train.npy, y_train.npy,
                    pw_val.npy, y_val.npy.
        models_dir: Directory where artefacts will be saved.
        config:     Optional BaselineConfig; defaults are used if None.

    Returns:
        dict with validation metrics and paths to saved artefacts.
    """
    if config is None:
        config = BaselineConfig()

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training split …")
    pw_train, y_train = _load_split(data_dir, "train")
    logger.info("Loading validation split …")
    pw_val, y_val = _load_split(data_dir, "val")

    # --- TF-IDF features ---
    extractor = CharNgramExtractor(config)
    X_train_tfidf = extractor.fit_transform(pw_train)
    X_val_tfidf = extractor.transform(pw_val)

    # --- Hand-crafted features ---
    logger.info("Extracting hand-crafted features (train) …")
    hc_train = extract_handcrafted_features(pw_train)
    logger.info("Extracting hand-crafted features (val) …")
    hc_val = extract_handcrafted_features(pw_val)

    # --- Combined feature matrix ---
    X_train = _build_feature_matrix(X_train_tfidf, hc_train)
    X_val = _build_feature_matrix(X_val_tfidf, hc_val)

    # --- Train ---
    clf = BaselineClassifier(config)
    metrics = clf.train(X_train, y_train, X_val, y_val)

    # --- Save ---
    extractor_path = models_dir / "baseline_extractor.joblib"
    model_path = models_dir / "baseline_model.joblib"
    extractor.save(extractor_path)
    clf.save(model_path)

    metrics["extractor_path"] = str(extractor_path)
    metrics["model_path"] = str(model_path)
    return metrics


def evaluate_baseline(data_dir: Path, models_dir: Path) -> dict:
    """Load saved baseline model and evaluate it on the test split.

    Args:
        data_dir:   Directory containing pw_test.npy, y_test.npy.
        models_dir: Directory containing saved artefacts.

    Returns:
        dict with full test-set metrics.
    """
    models_dir = Path(models_dir)

    extractor = CharNgramExtractor.load(models_dir / "baseline_extractor.joblib")
    clf = BaselineClassifier.load(models_dir / "baseline_model.joblib")

    logger.info("Loading test split …")
    pw_test, y_test = _load_split(data_dir, "test")

    X_test_tfidf = extractor.transform(pw_test)
    hc_test = extract_handcrafted_features(pw_test)
    X_test = _build_feature_matrix(X_test_tfidf, hc_test)

    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "precision_macro": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "per_class": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    logger.info(
        "Test results — acc=%.4f, F1-macro=%.4f",
        metrics["accuracy"],
        metrics["f1_macro"],
    )
    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline password-strength classifier (char n-gram TF-IDF + LR)"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing the processed data splits (.npy files).",
    )
    parser.add_argument(
        "--models_dir",
        type=Path,
        required=True,
        help="Directory where trained artefacts will be saved / loaded from.",
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="Skip training and only evaluate the already-saved model on the test set.",
    )
    args = parser.parse_args()

    if args.evaluate_only:
        metrics = evaluate_baseline(args.data_dir, args.models_dir)
        logger.info("Evaluation complete: %s", metrics)
    else:
        train_metrics = train_baseline(args.data_dir, args.models_dir)
        logger.info("Training complete. Val metrics: %s", train_metrics)
        test_metrics = evaluate_baseline(args.data_dir, args.models_dir)
        logger.info("Test metrics: %s", test_metrics)


if __name__ == "__main__":
    main()
