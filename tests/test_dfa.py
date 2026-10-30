"""
tests/test_dfa.py
-----------------
Unit tests for the JFLAP parser and DFA simulator.
Tests cover: parsing validity, all required transitions, acceptance/rejection
logic, boundary cases, malformed inputs, and dead-state handling.
"""

import os
import sys
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jflap_parser import parse_jff, describe
from dfa_simulator import DFASimulator, load_simulator, char_to_symbol, tokenize

JFF_PATH = os.path.join(os.path.dirname(__file__), "..", "automata", "password_dfa.jff")


class TestJFLAPParser(unittest.TestCase):
    def setUp(self):
        self.spec = parse_jff(JFF_PATH)

    def test_file_parses_without_error(self):
        self.assertIsNotNone(self.spec)

    def test_has_states(self):
        self.assertGreater(len(self.spec.states), 0)

    def test_has_initial_state(self):
        self.assertIn(self.spec.initial_state, self.spec.states)

    def test_has_accepting_states(self):
        self.assertGreater(len(self.spec.accepting_states), 0)
        for sid in self.spec.accepting_states:
            self.assertIn(sid, self.spec.states)

    def test_all_transitions_reference_valid_states(self):
        for (from_id, sym), to_id in self.spec.transitions.items():
            self.assertIn(from_id, self.spec.states,
                          f"Unknown source state {from_id}")
            self.assertIn(to_id, self.spec.states,
                          f"Unknown dest state {to_id}")

    def test_alphabet_symbols(self):
        # DFA must have at least U, L, D, S in its alphabet
        for sym in ["U", "L", "D", "S"]:
            self.assertIn(sym, self.spec.alphabet,
                          f"Symbol '{sym}' missing from alphabet")

    def test_describe_returns_string(self):
        desc = describe(self.spec)
        self.assertIsInstance(desc, str)
        self.assertGreater(len(desc), 0)


class TestCharMapping(unittest.TestCase):
    def test_uppercase_maps_to_U(self):
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            self.assertEqual(char_to_symbol(ch), "U")

    def test_lowercase_maps_to_L(self):
        for ch in "abcdefghijklmnopqrstuvwxyz":
            self.assertEqual(char_to_symbol(ch), "L")

    def test_digits_map_to_D(self):
        for ch in "0123456789":
            self.assertEqual(char_to_symbol(ch), "D")

    def test_special_maps_to_S(self):
        for ch in "!@#$%^&*()-_=+[]{}|;:',.<>?/`~\"\\":
            self.assertEqual(char_to_symbol(ch), "S",
                             f"Expected 'S' for char '{ch}'")

    def test_unicode_maps_to_X(self):
        for ch in ["é", "ñ", "中", "🔒"]:
            self.assertEqual(char_to_symbol(ch), "X")

    def test_tokenize_preserves_order(self):
        tokens = tokenize("Ab1!")
        self.assertEqual(tokens, ["U", "L", "D", "S"])


class TestDFASimulator(unittest.TestCase):
    def setUp(self):
        self.sim = load_simulator(JFF_PATH)

    # -- Acceptance --
    def test_valid_password_accepted(self):
        result = self.sim.run("Passw0rd!")
        self.assertTrue(result["accepted"], f"Expected ACCEPT, got: {result['violations']}")

    def test_all_categories_minimum_length(self):
        result = self.sim.run("Aa1!Aa1!")  # exactly 8 chars, all categories
        self.assertTrue(result["accepted"])

    def test_long_valid_password(self):
        result = self.sim.run("Correct#Horse9Battery")
        self.assertTrue(result["accepted"])

    # -- Rejection: missing categories --
    def test_missing_uppercase_rejected(self):
        result = self.sim.run("passw0rd!")
        self.assertFalse(result["accepted"])
        self.assertTrue(any("uppercase" in v for v in result["violations"]))

    def test_missing_lowercase_rejected(self):
        result = self.sim.run("PASSW0RD!")
        self.assertFalse(result["accepted"])
        self.assertTrue(any("lowercase" in v for v in result["violations"]))

    def test_missing_digit_rejected(self):
        result = self.sim.run("Password!")
        self.assertFalse(result["accepted"])
        self.assertTrue(any("digit" in v for v in result["violations"]))

    def test_missing_special_rejected(self):
        result = self.sim.run("Password1")
        self.assertFalse(result["accepted"])
        self.assertTrue(any("special" in v for v in result["violations"]))

    # -- Rejection: length --
    def test_exactly_7_chars_rejected(self):
        result = self.sim.run("Ab1!Ab1")  # 7 chars, all categories
        self.assertFalse(result["accepted"])
        self.assertTrue(any("Length" in v for v in result["violations"]))

    def test_empty_string_rejected(self):
        result = self.sim.run("")
        self.assertFalse(result["accepted"])

    # -- Boundary --
    def test_exactly_8_chars_all_categories(self):
        result = self.sim.run("Ab1!Cd2@")  # 8 chars, all categories
        self.assertTrue(result["accepted"])

    # -- Policy requirements dict --
    def test_policy_requirements_dict_correct(self):
        reqs = self.sim.check_policy_requirements("Passw0rd!")
        self.assertTrue(reqs["length_ok"])
        self.assertTrue(reqs["has_uppercase"])
        self.assertTrue(reqs["has_lowercase"])
        self.assertTrue(reqs["has_digit"])
        self.assertTrue(reqs["has_special"])

    def test_policy_requirements_detects_missing_digit(self):
        reqs = self.sim.check_policy_requirements("Password!")
        self.assertFalse(reqs["has_digit"])

    # -- Type safety --
    def test_non_string_raises_typeerror(self):
        with self.assertRaises(TypeError):
            self.sim.run(12345)

    # -- Missing file --
    def test_missing_jff_raises_file_not_found(self):
        from jflap_parser import parse_jff
        with self.assertRaises(FileNotFoundError):
            parse_jff("/nonexistent/path.jff")


class TestDFABoundaryCases(unittest.TestCase):
    def setUp(self):
        self.sim = load_simulator(JFF_PATH)

    def test_repeated_single_char_rejected(self):
        result = self.sim.run("aaaaaaaa")
        self.assertFalse(result["accepted"])

    def test_all_digits_rejected(self):
        result = self.sim.run("12345678")
        self.assertFalse(result["accepted"])

    def test_all_special_rejected(self):
        result = self.sim.run("!!!!!!!!!")
        self.assertFalse(result["accepted"])

    def test_very_long_password_accepted(self):
        # 64 chars with all categories
        pw = "Aa1!" * 16  # 64 chars
        result = self.sim.run(pw)
        self.assertTrue(result["accepted"])

    def test_unicode_in_otherwise_valid_password(self):
        # Unicode char is mapped to X -> handled gracefully (no crash)
        result = self.sim.run("Pässw0rd!")
        # Just verify no exception is raised and result is a dict
        self.assertIn("accepted", result)
        self.assertIsInstance(result["accepted"], bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
