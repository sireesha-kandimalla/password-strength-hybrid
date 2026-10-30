"""
src/app.py
----------
Streamlit web application for Password Strength Validation.
Provides 5 views:
  1. Live Hybrid Validator     – password input → SLM + DFA + Hybrid output.
  2. Model Evaluation Dashboard – test-set metrics, confusion matrices.
  3. DFA Inspection            – automaton states and transitions.
  4. Automated Tests           – run the test suite in-app.
  5. Architecture & Methodology – system explanation.

Security: Passwords are never stored, logged, or transmitted.
          Session state holds only the analysis result, not the plain password.
"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import streamlit as st

# Allow src imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
JFF_PATH = os.path.join(BASE_DIR, "automata", "password_dfa.jff")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Password Strength Validator — SLM + JFLAP DFA",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load artefacts (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading SLM and DFA…")
def load_system():
    from hybrid_engine import load_slm
    from dfa_simulator import load_simulator
    model, vocab, cfg, device = load_slm(MODEL_DIR)
    simulator = load_simulator(JFF_PATH)
    return model, vocab, cfg, device, simulator


@st.cache_data(show_spinner=False)
def load_eval_report():
    rpt_path = os.path.join(MODEL_DIR, "evaluation_report.json")
    if os.path.exists(rpt_path):
        with open(rpt_path) as f:
            return json.load(f)
    return None

# ── Sidebar navigation ────────────────────────────────────────────────────────
pages = [
    "🔒 Live Validator",
    "📊 Model Evaluation",
    "🤖 DFA Inspection",
    "🧪 Automated Tests",
    "📖 Architecture",
]
page = st.sidebar.radio("Navigate", pages)
st.sidebar.markdown("---")
st.sidebar.caption("All passwords are processed locally.\nNo data is stored or transmitted.")

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — Live Validator
# ═════════════════════════════════════════════════════════════════════════════
if page == "🔒 Live Validator":
    st.title("🔒 Password Strength Validator")
    st.markdown(
        "_Hybrid system: JFLAP DFA policy check + Small Language Model (Char Transformer)._"
    )
    st.markdown(
        "> **Security notice:** Your password is evaluated entirely on this machine. "
        "It is never saved, logged, or sent to any external service."
    )
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        show_pw = st.checkbox("Reveal password while typing", value=False)
        pw_type = "default" if show_pw else "password"
        password = st.text_input(
            "Enter a password to evaluate:",
            type=pw_type,
            max_chars=128,
            placeholder="Type your password here…",
            key="pw_input",
        )

    with col2:
        st.markdown("### Policy Requirements")
        st.markdown("""
- ✅ Minimum **8 characters**
- ✅ At least **one uppercase** letter `[A-Z]`
- ✅ At least **one lowercase** letter `[a-z]`
- ✅ At least **one digit** `[0-9]`
- ✅ At least **one special character** `(!@#$…)`
""")

    if password:
        try:
            model, vocab, cfg, device, simulator = load_system()
            from hybrid_engine import analyse
            result = analyse(password, model, vocab, device, simulator)

            # Clear password from local scope immediately
            del password

            verdict = result["composite_verdict"]
            status = result["overall_status"]
            slm = result["approach_a_slm"]
            dfa = result["approach_b_dfa"]
            hybrid = result["approach_c_hybrid"]

            # ── Status banner ──────────────────────────────────────────────
            st.markdown("---")
            status_colors = {
                "ACCEPT_ROBUST": "🟢",
                "ACCEPT_CONDITIONAL": "🟡",
                "REJECT_VULNERABILITY": "🔴",
                "REJECT_POLICY": "🔴",
            }
            icon = status_colors.get(verdict, "⚪")
            st.markdown(f"## {icon} Verdict: `{verdict}`")

            c1, c2, c3 = st.columns(3)

            # ── Approach A: SLM ────────────────────────────────────────────
            with c1:
                st.markdown("### 🧠 SLM Prediction")
                tier = slm["predicted_tier"]
                conf = slm["confidence"] * 100
                tier_colors = {
                    "Very Weak": "🔴", "Weak": "🟠",
                    "Medium": "🟡", "Strong": "🟢", "Very Strong": "🟢"
                }
                st.markdown(f"**{tier_colors.get(tier, '')} {tier}** — {conf:.1f}% confidence")
                st.markdown("**Class probabilities:**")
                probs_df = pd.DataFrame({
                    "Class": list(slm["class_probabilities"].keys()),
                    "Probability": list(slm["class_probabilities"].values()),
                })
                st.dataframe(probs_df.set_index("Class"), use_container_width=True)
                st.caption(f"Inference latency: {slm['latency_ms']} ms")

            # ── Approach B: DFA ────────────────────────────────────────────
            with c2:
                st.markdown("### 🤖 DFA Policy Check")
                dfa_ok = dfa["accepted"]
                st.markdown(f"**{'✅ COMPLIANT' if dfa_ok else '❌ NON-COMPLIANT'}**")
                st.markdown(f"Terminal state: `{dfa['final_state']}`")

                reqs = dfa["policy_requirements"]
                req_display = {
                    "Length ≥ 8": reqs["length_ok"],
                    "Has uppercase": reqs["has_uppercase"],
                    "Has lowercase": reqs["has_lowercase"],
                    "Has digit": reqs["has_digit"],
                    "Has special char": reqs["has_special"],
                }
                for req, ok in req_display.items():
                    icon_r = "✅" if ok else "❌"
                    st.markdown(f"{icon_r} {req}")

                if dfa["violations"]:
                    st.markdown("**Violations:**")
                    for v in dfa["violations"]:
                        st.warning(v)

            # ── Approach C: Hybrid ─────────────────────────────────────────
            with c3:
                st.markdown("### ⚖️ Hybrid Assessment")
                dis = hybrid["disagreement"]
                if dis["has_disagreement"]:
                    st.markdown(f"**Disagreement: `{dis['type']}`**")
                    st.info(dis["explanation"])
                else:
                    st.success("✅ SLM and DFA are in full agreement.")

                if hybrid["suggestions"]:
                    st.markdown("**Improvement Suggestions:**")
                    for s in hybrid["suggestions"]:
                        st.markdown(f"• {s}")

        except FileNotFoundError as e:
            st.error(f"Missing model artefact: {e}\n\nRun `python src/train_slm.py` first.")
        except Exception as e:
            st.error(f"Evaluation error: {e}")

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — Model Evaluation Dashboard
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Evaluation":
    st.title("📊 Model Evaluation Dashboard")
    st.markdown("Metrics measured on the **held-out test set** (1,500 samples, never used in training).")

    report = load_eval_report()
    if report is None:
        st.warning("Evaluation report not found. Run `python src/evaluate_models.py` to generate it.")
    else:
        CLASS_NAMES = ["Very Weak", "Weak", "Medium", "Strong", "Very Strong"]

        # ── Summary table ──────────────────────────────────────────────────
        st.markdown("### Comparison Table")
        rows = []
        for key, label in [("baseline", "Baseline (TF-IDF + LR)"), ("slm", "SLM (Char Transformer)")]:
            if key in report:
                m = report[key]
                rows.append({
                    "Model": label,
                    "Accuracy": m.get("accuracy", "N/A"),
                    "Macro-F1": m.get("macro_f1", "N/A"),
                    "Macro-Precision": m.get("macro_precision", "N/A"),
                    "Macro-Recall": m.get("macro_recall", "N/A"),
                    "Weak-PW Recall": m.get("weak_pw_recall", "N/A"),
                    "False-Strong Rate": m.get("false_strong_rate", "N/A"),
                    "Size (bytes)": m.get("model_size_bytes", "N/A"),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

        # ── Confusion matrices ──────────────────────────────────────────────
        col1, col2 = st.columns(2)
        for col, key, label in [(col1, "baseline", "Baseline"), (col2, "slm", "SLM")]:
            if key in report and "confusion_matrix" in report[key]:
                with col:
                    st.markdown(f"#### {label} Confusion Matrix")
                    cm = np.array(report[key]["confusion_matrix"])
                    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
                    st.dataframe(cm_df, use_container_width=True)

        # ── DFA stats ──────────────────────────────────────────────────────
        if "dfa" in report:
            st.markdown("### DFA Policy Statistics (Test Set)")
            dfa = report["dfa"]
            d1, d2, d3 = st.columns(3)
            d1.metric("Total Passwords", dfa["total"])
            d2.metric("Policy Compliant", f"{dfa['accepted']} ({100*dfa['accept_rate']:.1f}%)")
            d3.metric("Policy Rejected", dfa["rejected"])

        # ── Hybrid ablation ────────────────────────────────────────────────
        if "hybrid_ablation" in report:
            st.markdown("### Hybrid Ablation")
            abl = report["hybrid_ablation"]
            st.markdown(f"Disagreement rate: **{abl['disagreement_rate']*100:.1f}%**")
            vd = pd.DataFrame.from_dict(abl["verdict_distribution"], orient="index",
                                        columns=["Count"])
            st.dataframe(vd, use_container_width=True)

        # ── Latency ────────────────────────────────────────────────────────
        if "slm" in report and "latency_benchmark" in report["slm"]:
            st.markdown("### SLM Inference Latency")
            lat = report["slm"]["latency_benchmark"]
            l1, l2, l3 = st.columns(3)
            l1.metric("Mean", f"{lat['mean_ms']} ms")
            l2.metric("P95", f"{lat['p95_ms']} ms")
            l3.metric("P99", f"{lat['p99_ms']} ms")

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — DFA Inspection
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 DFA Inspection":
    st.title("🤖 JFLAP DFA Inspection")
    st.markdown(f"Automaton file: `{JFF_PATH}`")

    try:
        sys.path.insert(0, os.path.join(BASE_DIR, "src"))
        from jflap_parser import parse_jff, describe
        spec = parse_jff(JFF_PATH)
        st.success(f"✅ DFA parsed successfully — {len(spec.states)} states, {len(spec.transitions)} transitions.")

        st.markdown("### DFA Summary")
        st.code(describe(spec))

        st.markdown("### State Table")
        rows = []
        for sid, sname in spec.states.items():
            rows.append({
                "ID": sid, "Name": sname,
                "Initial": "✅" if sid == spec.initial_state else "",
                "Accepting": "✅" if sid in spec.accepting_states else "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("### Transition Table")
        t_rows = []
        for (from_id, sym), to_id in sorted(spec.transitions.items()):
            t_rows.append({
                "From": spec.states.get(from_id, from_id),
                "Symbol": sym,
                "To": spec.states.get(to_id, to_id),
            })
        st.dataframe(pd.DataFrame(t_rows), use_container_width=True)

        st.markdown("### Alphabet")
        st.markdown(
            "| Symbol | Category |\n|--------|----------|\n"
            "| **U** | Uppercase letter [A-Z] |\n"
            "| **L** | Lowercase letter [a-z] |\n"
            "| **D** | Digit [0-9] |\n"
            "| **S** | Special / punctuation |\n"
            "| **X** | Unicode / non-ASCII (treated as S if S in DFA alphabet) |"
        )
    except Exception as e:
        st.error(f"DFA load error: {e}")

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — Automated Tests
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Automated Tests":
    st.title("🧪 Automated Test Suite")
    tests_dir = os.path.join(BASE_DIR, "tests")

    if st.button("▶ Run All Tests Now"):
        with st.spinner("Running test suite…"):
            result = subprocess.run(
                ["python", "-m", "pytest", tests_dir, "-v", "--tb=short", "--no-header"],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                timeout=120,
            )
        if result.returncode == 0:
            st.success("✅ All tests passed!")
        else:
            st.error("❌ Some tests failed.")
        with st.expander("Full test output", expanded=True):
            st.code(result.stdout + result.stderr, language="text")

    st.markdown("### Test Files")
    for fname in ["test_dfa.py", "test_slm.py", "test_hybrid.py"]:
        fpath = os.path.join(tests_dir, fname)
        if os.path.exists(fpath):
            with st.expander(f"📄 {fname}"):
                with open(fpath) as f:
                    st.code(f.read(), language="python")

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — Architecture
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📖 Architecture":
    st.title("📖 System Architecture & Methodology")

    st.markdown("""
## Project: Password Strength Validation Using Finite Automata + SLM

### Overview
This system combines two fundamentally different assessment methods:

| Component | Role | Limitation |
|-----------|------|-----------|
| **JFLAP DFA** | Deterministic policy checker | Cannot assess statistical predictability |
| **SLM (Char Transformer)** | Learned strength predictor | Cannot verify rule compliance |
| **Hybrid Engine** | Arbitrates between both | Depends on training data quality |

> ⚠️ **Important**: DFA acceptance does not imply a password is secure or unpredictable.
> A password like `Password1!` satisfies every character rule but is a common breach pattern.

---

### SLM Architecture
```
Input password → Character tokenization → Padding to max_len=64
    → Embedding (d=128) + Sinusoidal Positional Encoding
    → 2× TransformerEncoderLayer (4 heads, FFN=256, dropout=0.2)
    → Masked Mean Pooling (ignores PAD tokens)
    → Linear(128→128) + ReLU + Dropout
    → Linear(128→5) → Softmax → Class probabilities
```

**Training details:**
- Loss: CrossEntropyLoss with inverse-frequency class weights
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)
- LR schedule: Cosine Annealing
- Early stopping: patience=5 epochs, metric=Macro-F1 on validation set
- Fixed random seed: 42

---

### DFA Architecture
The JFLAP DFA uses a **subset construction** to track 4 binary flags: {U, L, D, S}.
States are named `q{U}{L}{D}{S}` (e.g., `q1010` = saw uppercase and digit so far).
Length ≥ 8 is enforced by a sequential cascade: the DFA only reaches accepting states
after at least 8 character-consuming transitions.

**Alphabet:** Σ = {U, L, D, S}  (concrete characters are mapped to these categories)

---

### Hybrid Decision Matrix (2D Risk Arbitration)

|  | DFA ACCEPT | DFA REJECT |
|---|---|---|
| **SLM: Strong/Very Strong** | `ACCEPT_ROBUST` ✅ | `REJECT_POLICY` ❌ |
| **SLM: Medium** | `ACCEPT_CONDITIONAL` 🟡 | `REJECT_POLICY` ❌ |
| **SLM: Weak/Very Weak** | `REJECT_VULNERABILITY` ❌ | `REJECT_POLICY` ❌ |

The DFA has a **veto right**: non-compliance always leads to REJECT,
regardless of SLM confidence. This prevents bypassing explicit policies
with high-entropy but non-compliant passphrases.

---

### Security Properties
1. Passwords are processed **entirely in local memory**.
2. No passwords are written to disk, logs, or network sockets.
3. Training data labels are used for SLM supervision; raw passwords are discarded.
4. Session state stores only the analysis result, not the password.

---

### Limitations
- The SLM is trained on 10,000 stratified samples; real-world password distributions may differ.
- Character-level tokenization ignores semantic word boundaries.
- The DFA cannot detect dictionary words, common sequences, or keyboard walks.
- The SLM cannot guarantee detection of all adversarial or systematically constructed passwords.
- Neither component replaces proper password hashing, breach-database checks, or MFA.
""")

    # Show training history if available
    hist_path = os.path.join(MODEL_DIR, "slm_training_history.csv")
    if os.path.exists(hist_path):
        st.markdown("---")
        st.markdown("### SLM Training History")
        hist_df = pd.read_csv(hist_path)
        st.dataframe(hist_df, use_container_width=True)
