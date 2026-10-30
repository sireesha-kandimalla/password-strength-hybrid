# Academic Project Report: Password Strength Validation Using Finite Automata with JFLAP and an Improved Small Language Model (SLM)

**Project Title:** Password Strength Validation Using Finite Automata with JFLAP and an Improved Small Language Model (SLM)  
**Authors / System:** Hybrid Cybersecurity Engineering Group  
**Date:** October 2026  
**Repository Location:** `C:\Users\siree\.gemini\antigravity\scratch\password-strength-hybrid`

---

## Executive Summary

Password security remains a fundamental cornerstone of authentication. Conventional password validation relies on static regular expressions or character-set policies (length, uppercase, lowercase, digit, special character), which fail to detect predictable passwords (e.g., `P@ssw0rd123!`). Conversely, purely statistical or machine-learning approaches predict entropy and pattern strength but cannot guarantee adherence to mandatory organizational compliance standards.

This project designs, implements, and evaluates an end-to-end **Dual-Layer Hybrid Password Validation System** combining:
1. **Deterministic Finite Automaton (DFA)**: Formally constructed and validated in JFLAP 7.1 and executed via an XML parser and simulator in Python to enforce exact grammatical syntax ($L \subseteq \Sigma^*$).
2. **Small Language Model (SLM)**: A character-level Transformer encoder (2 layers, 4 attention heads, $d_{\text{model}}=128$, 299,525 parameters) that captures variable-length semantic dependencies, character repetitions, n-gram sequencing, and structural predictability across 5 strength tiers.
3. **2D Risk-Arbitration Hybrid Decision Engine**: Merges policy compliance and semantic strength into actionable verdicts with zero plain-text password exposure.

---

## 1. Dataset Inspection and Preprocessing

### 1.1 Dataset Schema & Distribution
The dataset source is the Kaggle Password Metrics Dataset (`password_metrics_dataset.csv`):
- **Raw Sample Count**: 10,000 samples.
- **Columns**: `password` (string), `strength` (continuous metric in $[0.0, 1.0]$).
- **Missing Values**: 0 nulls across all fields.
- **Duplicates**: Deduplicated based on SHA-256 password digests to prevent test leakage.

### 1.2 Categorization & Stratified Splitting
Continuous strength scores were discretized into 5 balanced ordinal tiers:
- Tier 0: `Very Weak` ($0.0 \le \text{score} < 0.2$)
- Tier 1: `Weak` ($0.2 \le \text{score} < 0.4$)
- Tier 2: `Medium` ($0.4 \le \text{score} < 0.6$)
- Tier 3: `Strong` ($0.6 \le \text{score} < 0.8$)
- Tier 4: `Very Strong` ($0.8 \le \text{score} \le 1.0$)

**Data Splits (Leakage-Resistant)**:
- **Training Set**: 7,000 samples (70.0%) — exactly 1,400 samples/class.
- **Validation Set**: 1,500 samples (15.0%) — exactly 300 samples/class.
- **Held-Out Test Set**: 1,500 samples (15.0%) — exactly 300 samples/class.
- **Random Seed**: Fixed at `42`.

---

## 2. Finite Automaton Design in JFLAP

### 2.1 Formal Policy Specification
The DFA enforces the standardized organizational policy:
$$\mathcal{P} = \{\text{Length} \ge 8\} \cap \{\ge 1 \text{ Upper}\} \cap \{\ge 1 \text{ Lower}\} \cap \{\ge 1 \text{ Digit}\} \cap \{\ge 1 \text{ Special}\}$$

### 2.2 Automaton Architecture ($\mathcal{M} = \langle Q, \Sigma, \delta, q_0, F \rangle$)
- **Alphabet ($\Sigma$)**: Categorical symbols $\Sigma = \{U, L, D, S\}$:
  - $U$: Uppercase ASCII letters `[A-Z]`
  - $L$: Lowercase ASCII letters `[a-z]`
  - $D$: Decimal digits `[0-9]`
  - $S$: Special ASCII symbols `[!@#$%^&*()_+-=[]{}|;':",./<>?`~\\]`
  - $X$: Out-of-alphabet / Unicode inputs (diverted to dead-state or treated as special)
- **State Space ($Q$)**: 16 subset-construction states represented by bitmask $b_3 b_2 b_1 b_0$:
  - Bit 3 ($8$): Uppercase encountered
  - Bit 2 ($4$): Lowercase encountered
  - Bit 1 ($2$): Decimal digit encountered
  - Bit 0 ($1$): Special symbol encountered
- **Initial State**: $q_0 = q_{0000}$ (State 0).
- **Accepting States ($F$)**: $F = \{q_{1111}\}$ (State 15).
- **Transition Function ($\delta$)**: Total transition function containing 64 transitions:
  $$\delta(q_i, \sigma) = q_{i \lor \text{bit}(\sigma)}$$
- **JFLAP File**: Exported as `automata/password_dfa.jff` in standard JFLAP 7.1 XML schema.

---

## 3. Small Language Model (SLM) Architecture

### 3.1 Model Formulation
A pure conventional classifier (e.g., Logistic Regression or Random Forest) cannot model variable-length sequence dependencies and positional context. Hence, a compact neural character Transformer was implemented:

$$\text{Embedding}(x_t) + \text{PositionalEncoding}(t) \longrightarrow \text{TransformerEncoderLayer} \times 2 \longrightarrow \text{MaskedMeanPool} \longrightarrow \text{MLP Head}$$

### 3.2 Hyperparameters & Configuration
- **Vocabulary Size**: 72 tokens (ASCII characters + `<PAD>` + `<UNK>`).
- **Embedding Dimension ($d_{\text{model}}$)**: 128.
- **Self-Attention Heads ($n_{\text{head}}$)**: 4 (head dimension = 32).
- **Feedforward Hidden Dimension ($d_{\text{ff}}$)**: 256.
- **Encoder Layers**: 2.
- **Dropout**: 0.20.
- **Max Sequence Length**: 64 tokens.
- **Total Trainable Parameters**: 299,525 (~1.21 MB on disk).
- **Optimizer**: AdamW ($\text{lr}=3\times 10^{-4}$, weight decay $10^{-4}$) with Cosine Annealing learning rate schedule.
- **Early Stopping**: Patience of 5 epochs monitoring validation Macro-F1.

---

## 4. Empirical Evaluation & Experimental Results

All models were evaluated on the identical held-out test set ($N=1,500$, 300 samples per class).

### 4.1 Comparative Model Performance

| Metric | Baseline (TF-IDF Char n-grams + LogReg) | Improved SLM (Char Transformer) | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 41.07% | **90.20%** | **+49.13% (+119.6% rel.)** |
| **Macro-Precision** | 39.56% | **90.22%** | **+50.66%** |
| **Macro-Recall** | 41.05% | **90.20%** | **+49.15%** |
| **Macro-F1 Score** | 39.94% | **90.17%** | **+50.23% (+125.8% rel.)** |
| **Weak-PW Recall ($C_0 + C_1$)** | 72.33% | **98.17%** | **+25.84%** |
| **False-Strong Rate** | 12.67% | **0.00%** | **-12.67% (eliminated)** |
| **Parameters / Size** | ~50,000 n-grams (1.46 MB) | 299,525 weights (1.21 MB) | **17.1% smaller storage** |
| **Inference Latency (Mean)** | 0.85 ms | 2.50 ms | Scalable real-time (<3ms) |
| **Inference Latency (P95)** | 1.10 ms | 3.46 ms | Responsive (<4ms) |

### 4.2 SLM Per-Class Performance Breakdown

| Class Tier | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Very Weak** | 0.8812 | 0.8870 | **0.8841** | 301 |
| **Weak** | 0.8780 | 0.8428 | **0.8601** | 299 |
| **Medium** | 0.9296 | 0.8800 | **0.9041** | 300 |
| **Strong** | 0.8707 | 0.9200 | **0.8947** | 300 |
| **Very Strong** | 0.9515 | 0.9800 | **0.9655** | 300 |
| **Macro Average** | **0.9022** | **0.9020** | **0.9017** | **1500** |

### 4.3 SLM Confusion Matrix (Rows: Ground Truth, Columns: Predicted)

$$\begin{pmatrix}
267 & 34 & 0 & 0 & 0 \\
36 & 252 & 11 & 0 & 0 \\
0 & 1 & 264 & 35 & 0 \\
0 & 0 & 9 & 276 & 15 \\
0 & 0 & 0 & 6 & 294
\end{pmatrix}$$

*Crucial Security Observation*: No off-diagonal errors exceed 1 tier step. Most importantly, **zero weak passwords ($C_0, C_1$) were predicted as Strong or Very Strong ($C_3, C_4$)**, yielding a **0.00% False-Strong Rate**.

---

## 5. DFA and Hybrid Arbitration Analysis

### 5.1 DFA Policy Coverage on Test Set
- **Total Test Samples**: 1,500
- **Policy Accepted**: 3 (0.2%)
- **Policy Rejected**: 1,497 (99.8%)
- **Interpretation**: Real-world and benchmark passwords heavily violate multi-category requirements, demonstrating why strict policy alone rejects 99.8% of user passwords without providing nuance regarding password strength.

### 5.2 Disagreement Analysis (SLM vs. DFA)
On the held-out test set, the SLM and DFA disagreed on **909 out of 1,500 cases (60.6%)**:
1. **High-Entropy but Non-Compliant (908 cases)**: Passwords rated Medium, Strong, or Very Strong by the SLM that lacked one specific symbol category or had 7 characters instead of 8 (e.g., long passphrases).
2. **Syntax-Compliant but Vulnerable (1 case)**: Passwords that satisfied all character classes (Upper + Lower + Digit + Special) but followed predictable dictionary roots or patterns (e.g., `Tr0ub4dor&3`).
3. **Robust Dual Agreement (2 cases)**: Satisfied all character requirements and scored Very Strong.

### 5.3 2D Risk-Arbitration Decision Matrix

| DFA Decision | SLM Strength Tier | Final Hybrid Verdict | Security Action |
| :---: | :---: | :---: | :--- |
| **ACCEPT** | Strong / Very Strong | `ACCEPT_ROBUST` | Full policy and entropy satisfaction. |
| **ACCEPT** | Medium | `ACCEPT_CONDITIONAL` | Compliant syntax; recommend minor entropy boost. |
| **ACCEPT** | Very Weak / Weak | `REJECT_VULNERABILITY` | **Blocked**: Syntax valid, but predictable pattern. |
| **REJECT** | Any Tier | `REJECT_POLICY` | **Blocked**: Mandatory policy criteria violated. |

---

## 6. Software Architecture & Verification

The project is structured into modular components:
- `automata/password_dfa.jff`: Verified JFLAP XML automaton.
- `src/data_preprocessing.py`: Leakage-safe stratified dataset pipeline.
- `src/train_baseline.py`: N-gram TF-IDF + Logistic Regression benchmark.
- `src/train_slm.py`: Character-level Transformer SLM training pipeline.
- `src/jflap_parser.py`: Direct XML parser (`DFASpec`, `parse_jff`, `describe`).
- `src/dfa_simulator.py`: Pure-Python DFA execution engine with category tokenization.
- `src/hybrid_engine.py`: Dual-model orchestration and 2D risk-arbitration inference.
- `src/evaluate_models.py`: Metric calculation, latency benchmarking, and report generation.
- `src/app.py`: Interactive Streamlit application with live validator and diagnostics.
- `tests/`: 72 automated unit and integration tests (`test_dfa.py`, `test_slm.py`, `test_hybrid.py`) passing with 100% success rate.

---

## 7. Limitations & Security Considerations

1. **Character Policy Limitations**: Satisfying character classes does not prevent dictionary attacks or compromised password reuse (e.g., HaveIBeenPwned breaches).
2. **Offline Dictionary Coverage**: While the SLM learns sub-word predictability, dedicated dictionary lookups should complement production pipelines.
3. **Data Privacy**: Passwords must never be stored in persistent logs or transmitted over unencrypted networks. All processing in this project is strictly local and ephemeral.
