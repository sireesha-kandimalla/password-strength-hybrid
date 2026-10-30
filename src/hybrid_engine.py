"""
hybrid_engine.py
----------------
Loads trained SLM and JFLAP DFA, runs Approach A (SLM-only), Approach B (DFA-only),
and Approach C (Hybrid SLM + DFA) with a transparent 2D Risk-Arbitration strategy.

Security: Passwords are never stored, logged, or transmitted. Input is consumed
in-memory and cleared after inference.

Hybrid Decision Matrix:
           DFA: ACCEPT     DFA: REJECT
SLM Strong  → ACCEPT_ROBUST    → REJECT_POLICY
SLM Weak    → REJECT_VULN      → REJECT_POLICY (dual failure)

Where 'strong' SLM = predicted tier in {'Medium','Strong','Very Strong'}.
"""

import os
import json
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional

# Allow src imports when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dfa_simulator import load_simulator, DFASimulator

# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES = ["Very Weak", "Weak", "Medium", "Strong", "Very Strong"]
STRONG_TIERS = {"Medium", "Strong", "Very Strong"}
PAD_IDX = 0
UNK_IDX = 1
MAX_LEN = 64


# ── SLM Loader ────────────────────────────────────────────────────────────────

class CharTransformerSLM(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model, nhead, num_layers,
                 dim_feedforward, dropout, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model, num_classes))
        self.max_len = max_len

    def forward(self, token_ids):
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(token_ids) + self.pos_enc(positions)
        padding_mask = (token_ids == PAD_IDX)
        all_padded = padding_mask.all(dim=-1)
        if all_padded.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_padded, 0] = False
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        mask = (~padding_mask).unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


def load_slm(model_dir: str):
    """Loads the best-checkpoint SLM and vocabulary from model_dir."""
    config_path = os.path.join(model_dir, "slm_config.json")
    vocab_path = os.path.join(model_dir, "char_vocab.json")
    weights_path = os.path.join(model_dir, "slm_best.pt")

    if not all(os.path.exists(p) for p in [config_path, vocab_path, weights_path]):
        raise FileNotFoundError(
            f"SLM artefacts missing in {model_dir}. "
            "Run src/train_slm.py first."
        )

    with open(config_path) as f:
        cfg = json.load(f)
    with open(vocab_path) as f:
        vocab = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharTransformerSLM(
        vocab_size=cfg["vocab_size"],
        num_classes=cfg["num_classes"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        max_len=cfg["max_len"],
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, vocab, cfg, device


# ── SLM Inference ─────────────────────────────────────────────────────────────

def encode_password(password: str, vocab: Dict[str, int], max_len: int = MAX_LEN):
    ids = [vocab.get(c, UNK_IDX) for c in password[:max_len]]
    ids += [PAD_IDX] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)


@torch.no_grad()
def slm_predict(password: str, model, vocab: dict, device) -> Dict[str, Any]:
    """Returns SLM strength prediction without logging the password."""
    t0 = time.perf_counter()
    tokens = encode_password(password, vocab).to(device)
    logits = model(tokens)
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "predicted_tier": CLASS_NAMES[pred_idx],
        "predicted_index": pred_idx,
        "confidence": float(probs[pred_idx]),
        "class_probabilities": {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)},
        "latency_ms": round(latency_ms, 2),
    }


# ── Hybrid Engine ─────────────────────────────────────────────────────────────

def _build_suggestions(dfa_result: Dict, slm_result: Dict) -> list:
    suggestions = []
    violations = dfa_result.get("violations", [])

    if not dfa_result["accepted"]:
        for v in violations:
            if "Length" in v:
                suggestions.append("Increase password length to at least 8 characters.")
            elif "uppercase" in v:
                suggestions.append("Add at least one uppercase letter [A-Z].")
            elif "lowercase" in v:
                suggestions.append("Add at least one lowercase letter [a-z].")
            elif "digit" in v:
                suggestions.append("Add at least one digit [0-9].")
            elif "special" in v:
                suggestions.append("Add at least one special character (e.g. !@#$%^&*).")

    tier = slm_result["predicted_tier"]
    if tier in ("Very Weak", "Weak"):
        suggestions.append("Avoid common dictionary words, names, or brand sequences.")
        suggestions.append("Avoid predictable substitutions (e.g. 'p@ssw0rd') and keyboard walks.")
        suggestions.append("Use randomized characters rather than memorable patterns.")
    elif tier == "Medium":
        suggestions.append("Consider lengthening the password with non-sequential characters.")
        suggestions.append("Mix character categories more unpredictably to reach 'Strong'.")

    return list(dict.fromkeys(suggestions))  # deduplicate preserving order


def _disagrement_analysis(dfa_accepted: bool, slm_tier: str) -> Dict:
    is_strong = slm_tier in STRONG_TIERS
    if dfa_accepted and not is_strong:
        return {
            "has_disagreement": True,
            "type": "SYNTAX_COMPLIANT_BUT_VULNERABLE",
            "explanation": (
                f"The password satisfies every JFLAP policy rule, but the SLM classified "
                f"it as '{slm_tier}' — indicating learnt patterns of predictability. "
                "A character-rule check alone cannot detect dictionary sequences or common patterns."
            ),
        }
    if not dfa_accepted and is_strong:
        return {
            "has_disagreement": True,
            "type": "HIGH_ENTROPY_BUT_NON_COMPLIANT",
            "explanation": (
                f"The SLM assessed this password as '{slm_tier}', suggesting strong entropy "
                "or length, but it was rejected by the DFA because it does not satisfy all "
                "required character-category rules. Add the missing character types to comply."
            ),
        }
    return {"has_disagreement": False, "type": "FULL_AGREEMENT", "explanation": None}


def hybrid_verdict(dfa_accepted: bool, slm_tier: str) -> str:
    """
    2D Risk-Arbitration Decision:
       REJECT_POLICY       – DFA rejected (policy violated).
       REJECT_VULNERABILITY – DFA accepted but SLM detected weak/predictable pattern.
       ACCEPT_CONDITIONAL  – DFA accepted + SLM says Medium (passes, needs improvement).
       ACCEPT_ROBUST       – DFA accepted + SLM says Strong or Very Strong.
    """
    if not dfa_accepted:
        return "REJECT_POLICY"
    if slm_tier in ("Very Weak", "Weak"):
        return "REJECT_VULNERABILITY"
    if slm_tier == "Medium":
        return "ACCEPT_CONDITIONAL"
    return "ACCEPT_ROBUST"


def analyse(password: str, model, vocab: dict, device, simulator: DFASimulator) -> Dict[str, Any]:
    """
    Runs all three approaches on a password and returns the combined assessment.
    The password string is used only within this function scope.
    """
    # Approach A – SLM
    slm_result = slm_predict(password, model, vocab, device)

    # Approach B – DFA
    dfa_result = simulator.run(password)
    policy_reqs = simulator.check_policy_requirements(password)

    # Approach C – Hybrid
    verdict = hybrid_verdict(dfa_result["accepted"], slm_result["predicted_tier"])
    overall = "ACCEPT" if verdict in ("ACCEPT_CONDITIONAL", "ACCEPT_ROBUST") else "REJECT"
    disagreement = _disagrement_analysis(dfa_result["accepted"], slm_result["predicted_tier"])
    suggestions = _build_suggestions(dfa_result, slm_result)

    return {
        "composite_verdict": verdict,
        "overall_status": overall,
        "approach_a_slm": slm_result,
        "approach_b_dfa": {
            "accepted": dfa_result["accepted"],
            "final_state": dfa_result["final_state"],
            "violations": dfa_result["violations"],
            "policy_requirements": policy_reqs,
        },
        "approach_c_hybrid": {
            "verdict": verdict,
            "status": overall,
            "disagreement": disagreement,
            "suggestions": suggestions,
        },
    }


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, pprint

    parser = argparse.ArgumentParser(description="Hybrid Engine CLI")
    parser.add_argument("password", nargs="?", help="Password to evaluate (for testing only)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models")
    jff_path = os.path.join(base_dir, "automata", "password_dfa.jff")

    print("Loading SLM...")
    model, vocab, cfg, device = load_slm(model_dir)
    print("Loading DFA simulator...")
    simulator = load_simulator(jff_path)
    print("Ready.\n")

    pw = args.password or "TestPwd9!"
    result = analyse(pw, model, vocab, device, simulator)
    # Print everything except the plain password
    pprint.pprint({k: v for k, v in result.items()})
