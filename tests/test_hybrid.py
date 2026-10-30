"""
tests/test_hybrid.py
--------------------
Integration tests for the hybrid engine:
- Correct composite verdicts per decision matrix.
- Disagreement detection.
- Suggestion generation.
- No password logging.
- Consistent SLM + DFA + hybrid output structure.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hybrid_engine import (
    load_slm, analyse, hybrid_verdict, _disagrement_analysis,
    _build_suggestions, CLASS_NAMES, STRONG_TIERS
)
from dfa_simulator import load_simulator

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "models")
JFF_PATH = os.path.join(BASE_DIR, "automata", "password_dfa.jff")


class TestHybridDecisionMatrix(unittest.TestCase):
    """Tests the 2D Risk-Arbitration decision function directly."""

    def test_dfa_reject_always_gives_reject_policy(self):
        for tier in CLASS_NAMES:
            result = hybrid_verdict(dfa_accepted=False, slm_tier=tier)
            self.assertEqual(result, "REJECT_POLICY")

    def test_dfa_accept_slm_very_weak_gives_reject_vulnerability(self):
        self.assertEqual(hybrid_verdict(True, "Very Weak"), "REJECT_VULNERABILITY")

    def test_dfa_accept_slm_weak_gives_reject_vulnerability(self):
        self.assertEqual(hybrid_verdict(True, "Weak"), "REJECT_VULNERABILITY")

    def test_dfa_accept_slm_medium_gives_accept_conditional(self):
        self.assertEqual(hybrid_verdict(True, "Medium"), "ACCEPT_CONDITIONAL")

    def test_dfa_accept_slm_strong_gives_accept_robust(self):
        self.assertEqual(hybrid_verdict(True, "Strong"), "ACCEPT_ROBUST")

    def test_dfa_accept_slm_very_strong_gives_accept_robust(self):
        self.assertEqual(hybrid_verdict(True, "Very Strong"), "ACCEPT_ROBUST")


class TestDisagreementAnalysis(unittest.TestCase):
    def test_compliant_but_weak_detects_syntax_compliant_vulnerable(self):
        result = _disagrement_analysis(dfa_accepted=True, slm_tier="Weak")
        self.assertTrue(result["has_disagreement"])
        self.assertEqual(result["type"], "SYNTAX_COMPLIANT_BUT_VULNERABLE")

    def test_strong_but_non_compliant_detects_high_entropy_non_compliant(self):
        result = _disagrement_analysis(dfa_accepted=False, slm_tier="Strong")
        self.assertTrue(result["has_disagreement"])
        self.assertEqual(result["type"], "HIGH_ENTROPY_BUT_NON_COMPLIANT")

    def test_full_agreement_accept_no_disagreement(self):
        result = _disagrement_analysis(dfa_accepted=True, slm_tier="Very Strong")
        self.assertFalse(result["has_disagreement"])
        self.assertEqual(result["type"], "FULL_AGREEMENT")

    def test_full_agreement_reject_no_disagreement(self):
        result = _disagrement_analysis(dfa_accepted=False, slm_tier="Very Weak")
        self.assertFalse(result["has_disagreement"])


class TestSuggestions(unittest.TestCase):
    def test_missing_uppercase_generates_suggestion(self):
        dfa_result = {"accepted": False, "violations": ["Missing at least one uppercase letter [A-Z]"]}
        slm_result = {"predicted_tier": "Weak"}
        suggestions = _build_suggestions(dfa_result, slm_result)
        self.assertTrue(any("uppercase" in s.lower() for s in suggestions))

    def test_missing_digit_generates_suggestion(self):
        dfa_result = {"accepted": False, "violations": ["Missing at least one digit [0-9]"]}
        slm_result = {"predicted_tier": "Medium"}
        suggestions = _build_suggestions(dfa_result, slm_result)
        self.assertTrue(any("digit" in s.lower() or "[0-9]" in s for s in suggestions))

    def test_weak_slm_adds_dictionary_suggestion(self):
        dfa_result = {"accepted": True, "violations": []}
        slm_result = {"predicted_tier": "Weak"}
        suggestions = _build_suggestions(dfa_result, slm_result)
        self.assertTrue(any("dictionary" in s.lower() for s in suggestions))

    def test_no_duplicate_suggestions(self):
        dfa_result = {"accepted": False, "violations": ["Missing at least one uppercase letter [A-Z]"]}
        slm_result = {"predicted_tier": "Very Weak"}
        suggestions = _build_suggestions(dfa_result, slm_result)
        self.assertEqual(len(suggestions), len(set(suggestions)))


class TestFullPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.vocab, cls.cfg, cls.device = load_slm(MODEL_DIR)
        cls.simulator = load_simulator(JFF_PATH)

    def _run(self, pw):
        return analyse(pw, self.model, self.vocab, self.device, self.simulator)

    def test_output_structure(self):
        result = self._run("Passw0rd!")
        for key in ["composite_verdict", "overall_status",
                    "approach_a_slm", "approach_b_dfa", "approach_c_hybrid"]:
            self.assertIn(key, result)

    def test_compliant_valid_verdict(self):
        result = self._run("Passw0rd!")
        self.assertIn(result["composite_verdict"],
                      ["ACCEPT_ROBUST", "ACCEPT_CONDITIONAL", "REJECT_VULNERABILITY"])

    def test_policy_violating_password_rejected(self):
        result = self._run("password")  # no upper, no digit, no special
        self.assertEqual(result["overall_status"], "REJECT")
        self.assertEqual(result["composite_verdict"], "REJECT_POLICY")

    def test_short_password_rejected_by_dfa(self):
        result = self._run("Ab1!")
        self.assertEqual(result["approach_b_dfa"]["accepted"], False)
        self.assertTrue(any("Length" in v for v in result["approach_b_dfa"]["violations"]))

    def test_slm_output_has_probabilities(self):
        result = self._run("TestPwd9!")
        probs = result["approach_a_slm"]["class_probabilities"]
        self.assertEqual(set(probs.keys()), set(CLASS_NAMES))

    def test_suggestions_is_list(self):
        result = self._run("short")
        self.assertIsInstance(result["approach_c_hybrid"]["suggestions"], list)

    def test_password_not_in_output(self):
        """Security test: the plain password must not appear in the output dict."""
        pw = "MyTestPwd9!"
        result = self._run(pw)
        result_str = str(result)
        self.assertNotIn(pw, result_str,
                         "SECURITY VIOLATION: plain password found in output")

    def test_empty_password_handled_gracefully(self):
        result = self._run("")
        self.assertEqual(result["overall_status"], "REJECT")

    def test_unicode_password_handled_gracefully(self):
        result = self._run("Pässw0rd!")
        self.assertIn("accepted", result["approach_b_dfa"])
        self.assertIn("predicted_tier", result["approach_a_slm"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
