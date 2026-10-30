# Password Strength Validation Using Finite Automata with JFLAP and an Improved Small Language Model (SLM)

A hybrid cybersecurity and machine learning framework combining formal language theory (Deterministic Finite Automata via JFLAP) and deep sequence modeling (Character Transformer SLM) for robust password strength assessment and compliance verification.

---

## Key Features

1. **Deterministic Policy Enforcement (DFA in JFLAP)**:
   - Formally designed in JFLAP 7.1 and exported as XML (`.jff`).
   - 16-state subset construction tracking mandatory character categories (Uppercase, Lowercase, Digits, Special Characters).
   - Zero-dependency Python simulation layer with total transition parsing.

2. **Semantic Pattern Modeling (Character-Level Transformer SLM)**:
   - 2-layer Transformer encoder with multi-head self-attention ($d_{\text{model}}=128$, 4 heads, $d_{\text{ff}}=256$, 299,525 parameters).
   - Predicts password strength across 5 ordinal tiers (`Very Weak`, `Weak`, `Medium`, `Strong`, `Very Strong`).
   - Achieves **90.20% Accuracy** and **90.17% Macro-F1** on the held-out test set, with a **0.00% False-Strong Rate**.

3. **2D Risk-Arbitration Hybrid Engine**:
   - Synthesizes grammatical compliance and empirical entropy into transparent, actionable verdicts.
   - Detects critical edge cases: *Syntax-Compliant but Vulnerable* (e.g., dictionary roots with required symbols) and *High-Entropy Non-Compliant* (e.g., long passphrases).

4. **Interactive Streamlit Web Dashboard**:
   - Live real-time validator with privacy toggle and zero-exposure password handling.
   - Diagnostic breakdown: satisfied/missing requirements, class probabilities, and suggestions.
   - Interactive evaluation dashboard with confusion matrices and performance metrics.

---

## Directory Structure

```
password-strength-hybrid/
├── automata/
│   └── password_dfa.jff        # Standard JFLAP 7.1 XML automaton file
├── data/
│   ├── raw/
│   │   └── password_metrics_dataset.csv
│   └── processed/
│       ├── train.csv           # 7,000 stratified training samples
│       ├── val.csv             # 1,500 stratified validation samples
│       ├── test.csv            # 1,500 held-out test samples
│       └── split_metadata.json
├── models/
│   ├── baseline_model.joblib   # TF-IDF + Logistic Regression benchmark
│   ├── slm_best.pt             # Trained Character Transformer weights
│   ├── slm_config.json         # SLM hyperparameter configuration
│   ├── char_vocab.json         # Character tokenizer vocabulary (72 tokens)
│   ├── slm_training_history.csv
│   └── evaluation_report.json  # Comprehensive empirical evaluation results
├── reports/
│   └── final_academic_report.md# Formal academic research report
├── src/
│   ├── data_preprocessing.py   # Stratified splitting and leakage prevention
│   ├── train_baseline.py       # TF-IDF + Logistic Regression training
│   ├── train_slm.py            # PyTorch Transformer SLM training loop
│   ├── jflap_parser.py         # JFLAP .jff XML parser and DFASpec definition
│   ├── dfa_simulator.py        # DFA simulation engine and violation generator
│   ├── hybrid_engine.py        # 2D Risk-Arbitration inference engine
│   ├── evaluate_models.py      # Test-set evaluation and latency benchmarking
│   └── app.py                  # Streamlit web user interface
├── tests/
│   ├── test_dfa.py             # Unit tests for JFLAP parsing and DFA simulation
│   ├── test_slm.py             # Unit tests for SLM loading, encoding, and inference
│   └── test_hybrid.py          # Integration tests for hybrid arbitration matrix
├── requirements.txt
└── README.md
```

---

## Quickstart & Execution Guide

### 1. Environment Setup
Install required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Automated Test Suite
Run all 72 unit and integration tests:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 3. Evaluate the Models on Held-Out Test Data
Compute accuracy, macro-F1, weak-password recall, false-strong rates, and latency benchmarks:
```bash
python src/evaluate_models.py
```

### 4. Launch the Interactive Streamlit Web Interface
Run the Streamlit application:
```bash
streamlit run src/app.py
```

---

## Experimental Benchmark Summary

| Metric | Baseline (TF-IDF + LogReg) | Improved SLM (Char Transformer) | Relative Gain |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 41.07% | **90.20%** | **+49.13% (+119.6%)** |
| **Macro-F1** | 39.94% | **90.17%** | **+50.23% (+125.8%)** |
| **Weak-PW Recall** | 72.33% | **98.17%** | **+25.84%** |
| **False-Strong Rate** | 12.67% | **0.00%** | **Eliminated (0%)** |
| **Model Size** | 1.46 MB | **1.21 MB** | **-17.1% smaller** |
| **P95 Latency** | 1.10 ms | **3.46 ms** | Real-time ready |
