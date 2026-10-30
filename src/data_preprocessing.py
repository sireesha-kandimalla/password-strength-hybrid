"""
data_preprocessing.py
=====================
Dataset inspection, preprocessing, and splitting for the password strength
classification project.

Dataset reference:
    https://www.kaggle.com/datasets/karthikudyawar/password-metrics-dataset

Security note
-------------
Raw password strings are NEVER printed, logged, or persisted beyond the
in-memory processing step.  Only derived features and character-index
sequences are saved to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------
DATA_DIR: Path = Path(__file__).parent.parent / "data"
MODELS_DIR: Path = Path(__file__).parent.parent / "models"

# Special vocab tokens
_PAD: int = 0
_UNK: int = 1
_SOS: int = 2
_EOS: int = 3

# ---------------------------------------------------------------------------
# 1. inspect_dataset
# ---------------------------------------------------------------------------

def inspect_dataset(filepath: Path) -> dict:
    """Load and inspect a CSV dataset, masking any password column.

    Parameters
    ----------
    filepath : Path
        Path to the CSV file.

    Returns
    -------
    dict
        Summary with keys: shape, dtypes, null_counts, duplicate_count,
        column_names, sample_count.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file cannot be parsed as CSV.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV at {filepath}: {exc}") from exc

    print(f"\n{'='*60}")
    print(f"Dataset: {filepath.name}")
    print(f"Shape  : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"{'='*60}")

    # Identify password-like columns so we never print raw values
    _pw_cols = _guess_password_columns(df)

    # ---- dtypes ----
    print("\n[Column dtypes]")
    print(df.dtypes.to_string())

    # ---- head (masked) ----
    print("\n[First 3 rows - passwords masked]")
    head = df.head(3).copy()
    for col in _pw_cols:
        head[col] = head[col].astype(str).apply(lambda s: "*" * len(s))
    print(head.to_string(index=False))

    # ---- null counts ----
    null_counts = df.isnull().sum()
    print("\n[Null counts per column]")
    print(null_counts[null_counts > 0].to_string() if null_counts.any() else "  None")

    # ---- duplicates ----
    dup_count = int(df.duplicated().sum())
    print(f"\n[Duplicate rows]: {dup_count:,}")

    # ---- unique values per column ----
    print("\n[Unique values per column]")
    for col in df.columns:
        n_unique = df[col].nunique()
        if col in _pw_cols:
            print(f"  {col:<30} {n_unique:>10,}  (password column - values masked)")
        else:
            print(f"  {col:<30} {n_unique:>10,}")

    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "null_counts": null_counts.to_dict(),
        "duplicate_count": dup_count,
        "column_names": list(df.columns),
        "sample_count": len(df),
    }
    return summary


def _guess_password_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that look like they store raw password strings."""
    keywords = ("password", "pass", "pwd")
    return [
        col for col in df.columns
        if any(kw in col.lower() for kw in keywords)
    ]


# ---------------------------------------------------------------------------
# 2. detect_columns
# ---------------------------------------------------------------------------

def detect_columns(df: pd.DataFrame) -> dict[str, Any]:
    """Heuristically detect the password, label, and numeric feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.

    Returns
    -------
    dict
        Keys: ``password_col`` (str | None), ``label_col`` (str | None),
        ``feature_cols`` (list[str]).
    """
    password_col: str | None = None
    label_col: str | None = None

    pw_keywords = ("password", "pass", "pwd")
    label_keywords = ("strength", "label", "class", "score", "target")

    for col in df.columns:
        col_lower = col.lower()
        if password_col is None and any(kw in col_lower for kw in pw_keywords):
            password_col = col
        if label_col is None and any(kw in col_lower for kw in label_keywords):
            label_col = col

    # Numeric feature columns: numeric dtype, not the label column
    feature_cols = [
        col for col in df.columns
        if col not in (password_col, label_col)
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    print(f"\n[Column detection]")
    print(f"  password_col  : {password_col!r}")
    print(f"  label_col     : {label_col!r}")
    print(f"  feature_cols  : {feature_cols}")

    return {
        "password_col": password_col,
        "label_col": label_col,
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------------
# 3. extract_char_features
# ---------------------------------------------------------------------------

def extract_char_features(password: str) -> dict[str, float]:
    """Derive numerical character-level features from a single password.

    The raw password string is consumed in-memory only; it is never stored
    or returned.

    Parameters
    ----------
    password : str
        The password to analyse.

    Returns
    -------
    dict[str, float]
        Feature name to value mapping.  Keys:

        length, num_upper, num_lower, num_digits, num_special, num_spaces,
        has_upper, has_lower, has_digit, has_special, unique_chars,
        max_consecutive_same, entropy.
    """
    if not isinstance(password, str):
        password = str(password)

    length = len(password)
    num_upper = sum(1 for c in password if c.isupper())
    num_lower = sum(1 for c in password if c.islower())
    num_digits = sum(1 for c in password if c.isdigit())
    num_spaces = sum(1 for c in password if c == " ")
    num_special = length - num_upper - num_lower - num_digits - num_spaces
    unique_chars = len(set(password))

    # Maximum run of the same character
    max_consec = 1
    cur_consec = 1
    for i in range(1, length):
        if password[i] == password[i - 1]:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 1
    if length == 0:
        max_consec = 0

    # Shannon entropy
    entropy = _shannon_entropy(password)

    return {
        "length": float(length),
        "num_upper": float(num_upper),
        "num_lower": float(num_lower),
        "num_digits": float(num_digits),
        "num_special": float(num_special),
        "num_spaces": float(num_spaces),
        "has_upper": float(num_upper > 0),
        "has_lower": float(num_lower > 0),
        "has_digit": float(num_digits > 0),
        "has_special": float(num_special > 0),
        "unique_chars": float(unique_chars),
        "max_consecutive_same": float(max_consec),
        "entropy": entropy,
    }


def _shannon_entropy(s: str) -> float:
    """Compute Shannon entropy (bits) for string *s*."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# ---------------------------------------------------------------------------
# 4. preprocess_dataset
# ---------------------------------------------------------------------------

def preprocess_dataset(
    df: pd.DataFrame,
    cols: dict[str, Any],
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """Deduplicate, featurise, encode labels, and split the dataset.

    Hash-based deduplication ensures raw passwords are never stored outside
    the in-memory DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (may contain raw password and label columns).
    cols : dict
        Output of :func:`detect_columns`.
    test_size : float
        Fraction of data to hold out for testing.
    val_size : float
        Fraction of *remaining* data (after test split) for validation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_val, X_test,
         y_train, y_val, y_test,
         pw_train, pw_val, pw_test,
         label_encoder, feature_names, vocab)

    Notes
    -----
    ``pw_*`` arrays contain character-index sequences (int arrays), NOT raw
    passwords.
    """
    password_col = cols.get("password_col")
    label_col = cols.get("label_col")
    extra_feature_cols = cols.get("feature_cols", [])

    if password_col is None:
        raise ValueError(
            "No password column detected. Provide column names manually via `cols`."
        )
    if label_col is None:
        raise ValueError(
            "No label column detected. Provide column names manually via `cols`."
        )

    df = df.copy()

    # ---- Drop rows with missing password or label ----
    before = len(df)
    df = df.dropna(subset=[password_col, label_col])
    print(f"\n[Preprocessing] Dropped {before - len(df):,} rows with null password/label.")

    # ---- Hash-based deduplication ----
    df["_pw_hash"] = df[password_col].astype(str).apply(
        lambda p: hashlib.sha256(p.encode("utf-8", errors="replace")).hexdigest()
    )
    before_dup = len(df)
    df = df.drop_duplicates(subset=["_pw_hash"])
    print(f"[Preprocessing] Removed {before_dup - len(df):,} duplicate passwords (hash-based).")

    # ---- Extract character features ----
    print("[Preprocessing] Extracting character features ...")
    char_feat_list = df[password_col].astype(str).apply(extract_char_features).tolist()
    char_feat_df = pd.DataFrame(char_feat_list, index=df.index)

    # ---- Combine with any existing numeric feature columns ----
    if extra_feature_cols:
        feature_df = pd.concat(
            [char_feat_df, df[extra_feature_cols].reset_index(drop=True)],
            axis=1,
        )
    else:
        feature_df = char_feat_df
    feature_names: list[str] = list(feature_df.columns)

    # ---- Encode labels ----
    labels_raw = df[label_col].astype(str).str.strip().str.lower()
    label_encoder = LabelEncoder()

    # Map common string labels to canonical ordering before encoding
    _str_map = {
        "weak": "0_weak",
        "medium": "1_medium",
        "moderate": "1_medium",
        "strong": "2_strong",
        "very strong": "3_verystrong",
        "verystrong": "3_verystrong",
    }
    labels_mapped = labels_raw.map(lambda v: _str_map.get(v, v))
    label_encoder.fit(labels_mapped)
    y_encoded = label_encoder.transform(labels_mapped)

    # ---- Class distribution ----
    unique_classes, counts = np.unique(y_encoded, return_counts=True)
    print("\n[Class distribution]")
    for cls, cnt in zip(unique_classes, counts):
        class_name = label_encoder.classes_[cls]
        print(f"  Class {cls} ({class_name}): {cnt:,} samples ({cnt/len(y_encoded)*100:.1f}%)")

    X = feature_df.values.astype(np.float32)
    y = y_encoded.astype(np.int64)

    # ---- Stratified train / test split ----
    X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
        X, y, np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # ---- Stratified train / val split ----
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train_val, y_train_val, idx_train_val,
        test_size=val_frac,
        random_state=random_state,
        stratify=y_train_val,
    )

    print(
        f"\n[Split sizes]  train={len(y_train):,}  val={len(y_val):,}  test={len(y_test):,}"
    )

    # ---- Build character vocabulary from TRAINING passwords only ----
    df_reset = df.reset_index(drop=True)
    train_passwords = df_reset.iloc[idx_train][password_col].astype(str).tolist()
    val_passwords   = df_reset.iloc[idx_val][password_col].astype(str).tolist()
    test_passwords  = df_reset.iloc[idx_test][password_col].astype(str).tolist()

    vocab = build_char_vocab(train_passwords)

    # ---- Convert passwords to index sequences ----
    print("[Preprocessing] Converting passwords to character-index sequences ...")
    pw_train = np.array(
        [password_to_indices(p, vocab) for p in train_passwords], dtype=np.int32
    )
    pw_val = np.array(
        [password_to_indices(p, vocab) for p in val_passwords], dtype=np.int32
    )
    pw_test = np.array(
        [password_to_indices(p, vocab) for p in test_passwords], dtype=np.int32
    )

    # Raw password strings are no longer needed — explicitly del for clarity
    del train_passwords, val_passwords, test_passwords

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        pw_train, pw_val, pw_test,
        label_encoder, feature_names, vocab,
    )


# ---------------------------------------------------------------------------
# 5. build_char_vocab
# ---------------------------------------------------------------------------

def build_char_vocab(passwords_train: list[str]) -> dict[str, int]:
    """Build a character vocabulary from training-set passwords only.

    Special tokens are prepended so their indices are fixed:
    ``<PAD>=0``, ``<UNK>=1``, ``<SOS>=2``, ``<EOS>=3``.

    Parameters
    ----------
    passwords_train : list[str]
        Raw password strings from the **training set only**.

    Returns
    -------
    dict[str, int]
        Mapping ``character -> integer index``.
    """
    char_counts: Counter = Counter()
    for pw in passwords_train:
        char_counts.update(str(pw))

    # Sort by frequency descending for a stable, reproducible vocab
    chars_sorted = sorted(char_counts.keys(), key=lambda c: -char_counts[c])

    vocab: dict[str, int] = {
        "<PAD>": _PAD,
        "<UNK>": _UNK,
        "<SOS>": _SOS,
        "<EOS>": _EOS,
    }
    next_idx = 4
    for ch in chars_sorted:
        if ch not in vocab:
            vocab[ch] = next_idx
            next_idx += 1

    print(f"[Vocab] Size: {len(vocab)} tokens (including 4 special tokens).")
    return vocab


# ---------------------------------------------------------------------------
# 6. password_to_indices
# ---------------------------------------------------------------------------

def password_to_indices(
    password: str,
    vocab: dict[str, int],
    max_len: int = 64,
) -> list[int]:
    """Convert a password string to a padded list of token indices.

    The sequence layout is:
        [<SOS>, c_0, c_1, ..., c_{min(len, max_len-2)-1}, <EOS>, <PAD>, ...]

    Total length is always ``max_len``.

    Parameters
    ----------
    password : str
        The password to encode.
    vocab : dict[str, int]
        Character to index mapping (output of :func:`build_char_vocab`).
    max_len : int
        Fixed sequence length (truncate or pad to this size).

    Returns
    -------
    list[int]
        Integer indices of length ``max_len``.
    """
    # Reserve positions for <SOS> and <EOS>; usable chars = max_len - 2
    usable = max_len - 2
    chars = list(str(password))[:usable]
    indices = (
        [_SOS]
        + [vocab.get(c, _UNK) for c in chars]
        + [_EOS]
    )
    # Pad to max_len
    indices += [_PAD] * (max_len - len(indices))
    return indices[:max_len]


# ---------------------------------------------------------------------------
# 7. save_splits
# ---------------------------------------------------------------------------

def save_splits(
    output_dir: Path,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    pw_train: np.ndarray,
    pw_val: np.ndarray,
    pw_test: np.ndarray,
    vocab: dict[str, int],
    label_encoder: LabelEncoder,
    feature_names: list[str],
) -> None:
    """Persist all split arrays and metadata to *output_dir*.

    Raw password strings are NEVER written to disk.

    Parameters
    ----------
    output_dir : Path
        Directory where split files will be saved (created if absent).
    X_train, X_val, X_test : np.ndarray
        Numeric feature matrices.
    y_train, y_val, y_test : np.ndarray
        Encoded label arrays.
    pw_train, pw_val, pw_test : np.ndarray
        Character-index sequence arrays (shape: [n_samples, max_len]).
    vocab : dict[str, int]
        Character vocabulary.
    label_encoder : LabelEncoder
        Fitted sklearn LabelEncoder.
    feature_names : list[str]
        Names of the numeric features (columns of X_*).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "pw_train": pw_train,
        "pw_val": pw_val,
        "pw_test": pw_test,
    }
    for name, arr in arrays.items():
        np.save(output_dir / f"{name}.npy", arr)
        print(f"  Saved {name}.npy  shape={arr.shape}  dtype={arr.dtype}")

    joblib.dump(vocab, output_dir / "vocab.joblib")
    print(f"  Saved vocab.joblib  ({len(vocab)} tokens)")

    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")
    print(f"  Saved label_encoder.joblib  classes={list(label_encoder.classes_)}")

    with open(output_dir / "feature_names.yaml", "w", encoding="utf-8") as fh:
        yaml.dump({"feature_names": feature_names}, fh, allow_unicode=True)
    print(f"  Saved feature_names.yaml  ({len(feature_names)} features)")


# ---------------------------------------------------------------------------
# 8. load_splits
# ---------------------------------------------------------------------------

def load_splits(output_dir: Path) -> dict[str, Any]:
    """Load all split arrays and metadata saved by :func:`save_splits`.

    Parameters
    ----------
    output_dir : Path
        Directory containing the saved split files.

    Returns
    -------
    dict
        Keys: X_train, X_val, X_test, y_train, y_val, y_test,
              pw_train, pw_val, pw_test, vocab, label_encoder, feature_names.

    Raises
    ------
    FileNotFoundError
        If *output_dir* or required files are missing.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {output_dir}")

    result: dict[str, Any] = {}

    array_keys = [
        "X_train", "X_val", "X_test",
        "y_train", "y_val", "y_test",
        "pw_train", "pw_val", "pw_test",
    ]
    for key in array_keys:
        fpath = output_dir / f"{key}.npy"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing split file: {fpath}")
        result[key] = np.load(fpath)

    vocab_path = output_dir / "vocab.joblib"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocab file: {vocab_path}")
    result["vocab"] = joblib.load(vocab_path)

    le_path = output_dir / "label_encoder.joblib"
    if not le_path.exists():
        raise FileNotFoundError(f"Missing label encoder file: {le_path}")
    result["label_encoder"] = joblib.load(le_path)

    fn_path = output_dir / "feature_names.yaml"
    if not fn_path.exists():
        raise FileNotFoundError(f"Missing feature_names file: {fn_path}")
    with open(fn_path, "r", encoding="utf-8") as fh:
        result["feature_names"] = yaml.safe_load(fh).get("feature_names", [])

    print(f"[load_splits] Loaded all splits from: {output_dir}")
    return result


# ---------------------------------------------------------------------------
# 9. main - CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line interface for dataset preprocessing.

    Usage
    -----
    ::

        python data_preprocessing.py --data_path /path/to/passwords.csv
                                      [--output_dir /path/to/splits]
                                      [--test_size 0.15]
                                      [--val_size 0.15]
                                      [--random_state 42]
                                      [--max_len 64]
    """
    parser = argparse.ArgumentParser(
        description="Preprocess password strength dataset."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DATA_DIR / "splits",
        help="Directory where processed splits are saved (default: data/splits).",
    )
    parser.add_argument("--test_size",    type=float, default=0.15)
    parser.add_argument("--val_size",     type=float, default=0.15)
    parser.add_argument("--random_state", type=int,   default=42)
    parser.add_argument("--max_len",      type=int,   default=64)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(" Password Strength - Data Preprocessing")
    print(f"{'='*60}")
    print(f"  data_path    : {args.data_path}")
    print(f"  output_dir   : {args.output_dir}")
    print(f"  test_size    : {args.test_size}")
    print(f"  val_size     : {args.val_size}")
    print(f"  random_state : {args.random_state}")
    print(f"  max_len      : {args.max_len}")

    # -- Step 1: Inspect --
    try:
        summary = inspect_dataset(args.data_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] inspect_dataset failed: {exc}")
        raise SystemExit(1)

    # -- Step 2: Load and detect columns --
    try:
        df = pd.read_csv(args.data_path, low_memory=False)
        cols = detect_columns(df)
    except Exception as exc:
        print(f"\n[ERROR] detect_columns failed: {exc}")
        raise SystemExit(1)

    # -- Step 3: Preprocess --
    try:
        (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            pw_train, pw_val, pw_test,
            label_encoder, feature_names, vocab,
        ) = preprocess_dataset(
            df,
            cols,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
        )
    except (ValueError, KeyError) as exc:
        print(f"\n[ERROR] preprocess_dataset failed: {exc}")
        raise SystemExit(1)

    # -- Step 4: Save --
    try:
        print(f"\n[Saving splits to: {args.output_dir}]")
        save_splits(
            output_dir=args.output_dir,
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            pw_train=pw_train, pw_val=pw_val, pw_test=pw_test,
            vocab=vocab,
            label_encoder=label_encoder,
            feature_names=feature_names,
        )
    except Exception as exc:
        print(f"\n[ERROR] save_splits failed: {exc}")
        raise SystemExit(1)

    # -- Summary report --
    print(f"\n{'='*60}")
    print(" Summary Report")
    print(f"{'='*60}")
    print(f"  Dataset shape    : {summary['shape']}")
    print(f"  Columns          : {summary['column_names']}")
    print(f"  Duplicates found : {summary['duplicate_count']:,}")
    print(f"  Train samples    : {len(y_train):,}")
    print(f"  Val   samples    : {len(y_val):,}")
    print(f"  Test  samples    : {len(y_test):,}")
    print(f"  Feature count    : {len(feature_names)}")
    print(f"  Vocab size       : {len(vocab)}")
    print(f"  Label classes    : {list(label_encoder.classes_)}")
    print(f"\n  Splits saved to  : {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
