"""
tests/test_slm.py
-----------------
Unit tests for the SLM: loading, vocabulary, encoding, inference shape,
probability normalization, and output format.
"""

import os
import sys
import unittest
import torch
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hybrid_engine import load_slm, slm_predict, encode_password, CLASS_NAMES

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "models")


class TestSLMLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.vocab, cls.cfg, cls.device = load_slm(MODEL_DIR)

    def test_model_loads(self):
        self.assertIsNotNone(self.model)

    def test_vocab_has_pad_and_unk(self):
        self.assertIn("<PAD>", self.vocab)
        self.assertIn("<UNK>", self.vocab)
        self.assertEqual(self.vocab["<PAD>"], 0)
        self.assertEqual(self.vocab["<UNK>"], 1)

    def test_config_has_required_keys(self):
        required = ["vocab_size", "num_classes", "d_model", "nhead", "num_layers",
                    "dim_feedforward", "dropout", "max_len"]
        for key in required:
            self.assertIn(key, self.cfg, f"Config missing key: {key}")

    def test_model_in_eval_mode(self):
        self.assertFalse(self.model.training, "Model should be in eval mode after loading")


class TestSLMEncoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.vocab, cls.cfg, _ = load_slm(MODEL_DIR)

    def test_encode_produces_correct_length(self):
        tokens = encode_password("Passw0rd!", self.vocab)
        self.assertEqual(tokens.shape[1], self.cfg["max_len"])

    def test_encode_pads_short_password(self):
        tokens = encode_password("Ab1!", self.vocab)
        # Token IDs after position 4 should be PAD (0)
        self.assertTrue((tokens[0, 4:] == 0).all())

    def test_encode_truncates_long_password(self):
        long_pw = "A" * 200
        tokens = encode_password(long_pw, self.vocab, max_len=64)
        self.assertEqual(tokens.shape[1], 64)

    def test_unknown_chars_get_unk_index(self):
        tokens = encode_password("中文", self.vocab)
        unk_idx = self.vocab["<UNK>"]
        # Both chars should be UNK
        self.assertEqual(tokens[0, 0].item(), unk_idx)
        self.assertEqual(tokens[0, 1].item(), unk_idx)


class TestSLMInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.vocab, cls.cfg, cls.device = load_slm(MODEL_DIR)

    def _predict(self, pw: str):
        return slm_predict(pw, self.model, self.vocab, self.device)

    def test_output_has_required_keys(self):
        result = self._predict("Passw0rd!")
        for key in ["predicted_tier", "predicted_index", "confidence",
                    "class_probabilities", "latency_ms"]:
            self.assertIn(key, result)

    def test_predicted_tier_is_valid_class(self):
        result = self._predict("Passw0rd!")
        self.assertIn(result["predicted_tier"], CLASS_NAMES)

    def test_probabilities_sum_to_one(self):
        result = self._predict("Passw0rd!")
        total = sum(result["class_probabilities"].values())
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_all_classes_present_in_output(self):
        result = self._predict("Passw0rd!")
        self.assertEqual(set(result["class_probabilities"].keys()), set(CLASS_NAMES))

    def test_confidence_matches_max_probability(self):
        result = self._predict("Passw0rd!")
        max_prob = max(result["class_probabilities"].values())
        self.assertAlmostEqual(result["confidence"], max_prob, places=3)

    def test_latency_is_positive(self):
        result = self._predict("test123")
        self.assertGreater(result["latency_ms"], 0)

    def test_empty_string_does_not_crash(self):
        # Empty password is valid input for inference (SLM classifies it)
        result = self._predict("")
        self.assertIn("predicted_tier", result)

    def test_deterministic_across_calls(self):
        # Same password should always give same prediction in eval mode
        r1 = self._predict("MyPassword1!")
        r2 = self._predict("MyPassword1!")
        self.assertEqual(r1["predicted_tier"], r2["predicted_tier"])
        self.assertAlmostEqual(r1["confidence"], r2["confidence"], places=4)

    def test_known_very_weak_password(self):
        result = self._predict("1234")
        # Should predict Very Weak or Weak (not Strong/Very Strong)
        self.assertNotIn(result["predicted_tier"], ["Strong", "Very Strong"],
                         "Short trivial password should not be classified as Strong")


if __name__ == "__main__":
    unittest.main(verbosity=2)
