"""
dfa_simulator.py
----------------
Pure-Python simulator for the JFLAP DFA exported as a .jff file.
Reproduces acceptance / rejection exactly as JFLAP would on the same input.

Character-to-symbol mapping (alphabet Σ = {U, L, D, S, X}):
    U  =  Uppercase letter    [A-Z]
    L  =  Lowercase letter    [a-z]
    D  =  Digit               [0-9]
    S  =  Special character   (all printable ASCII not in U/L/D)
    X  =  Unknown / Unicode   (anything outside standard printable ASCII)

The DFA encodes 16 category-tracking states (subset construction over {U,L,D,S})
plus a length-enforcement cascade. The simulator maps each concrete character to
one of the five category symbols before feeding it to the transition function.
"""

import os
import string
from typing import Dict, List, Optional, Set, Tuple

from jflap_parser import DFASpec, parse_jff, describe

# ── Character-to-symbol mapping ───────────────────────────────────────────────

_UPPERCASE = set(string.ascii_uppercase)
_LOWERCASE = set(string.ascii_lowercase)
_DIGITS = set(string.digits)
_SPECIAL = set(string.punctuation + " ")
_PRINTABLE = _UPPERCASE | _LOWERCASE | _DIGITS | _SPECIAL


def char_to_symbol(ch: str) -> str:
    """
    Maps a single character to the DFA alphabet symbol.

    Returns:
        'U'  if the character is an uppercase ASCII letter.
        'L'  if the character is a lowercase ASCII letter.
        'D'  if the character is an ASCII digit.
        'S'  if the character is a printable ASCII special character.
        'X'  for any Unicode or non-printable character not in the above.
    """
    if ch in _UPPERCASE:
        return "U"
    if ch in _LOWERCASE:
        return "L"
    if ch in _DIGITS:
        return "D"
    if ch in _SPECIAL:
        return "S"
    return "X"


def tokenize(password: str) -> List[str]:
    """Returns the symbol sequence for the given password string."""
    return [char_to_symbol(c) for c in password]


# ── Simulation logic ──────────────────────────────────────────────────────────

class DFASimulator:
    """
    Simulates a parsed DFA from a JFLAP .jff file.

    Handles:
    - Missing transitions (treated as implicit rejection / dead state).
    - The 'X' category (mapped to 'S' if 'S' is in alphabet, else dead state).
    - Malformed input (zero-length, non-string).
    """

    DEAD_STATE = "__dead__"

    def __init__(self, spec: DFASpec, jff_path: Optional[str] = None):
        self.spec = spec
        self.jff_path = jff_path
        # If DFA has an 'S' transition, unknown chars are treated as special.
        self._x_maps_to_s = "S" in spec.alphabet

    def _resolve_symbol(self, sym: str) -> str:
        """Returns the resolved DFA alphabet symbol, or None for dead-state."""
        if sym in self.spec.alphabet:
            return sym
        if sym == "X" and self._x_maps_to_s:
            return "S"
        return None  # dead transition

    def run(self, password: str) -> Dict:
        """
        Runs the DFA on the given password.

        Returns a dict with:
            accepted      (bool)   – whether the password ends in an accepting state.
            final_state   (str)    – name of the terminal state.
            final_state_id (str)   – id of the terminal state.
            trace         (list)   – sequence of (symbol, state_name) pairs.
            violations    (list)   – human-readable list of failed requirements.
            dead_state    (bool)   – True if a missing transition was encountered.
        """
        if not isinstance(password, str):
            raise TypeError(f"Expected str, got {type(password).__name__}")

        symbols = tokenize(password)
        current = self.spec.initial_state
        trace = [(None, self.spec.states.get(current, current))]
        dead = False

        for sym in symbols:
            resolved = self._resolve_symbol(sym)
            if resolved is None:
                dead = True
                current = self.DEAD_STATE
                break
            key = (current, resolved)
            if key not in self.spec.transitions:
                dead = True
                current = self.DEAD_STATE
                break
            current = self.spec.transitions[key]
            trace.append((sym, self.spec.states.get(current, current)))

        accepted = (not dead) and (current in self.spec.accepting_states) and (len(password) >= 8)
        final_name = "dead_state" if dead else self.spec.states.get(current, current)

        violations = self._compute_violations(password, current, dead)

        return {
            "accepted": accepted,
            "final_state": final_name,
            "final_state_id": current if not dead else self.DEAD_STATE,
            "trace": trace,
            "violations": violations,
            "dead_state": dead,
        }

    def _compute_violations(self, password: str, final_id: str, dead: bool) -> List[str]:
        """
        Derives human-readable policy violations from the password.
        This mirrors the DFA's category-tracking logic in natural language.
        """
        violations = []
        if len(password) < 8:
            violations.append(f"Length violation: Length is {len(password)}, required ≥ 8")
        if not any(c in _UPPERCASE for c in password):
            violations.append("Missing at least one uppercase letter [A-Z]")
        if not any(c in _LOWERCASE for c in password):
            violations.append("Missing at least one lowercase letter [a-z]")
        if not any(c in _DIGITS for c in password):
            violations.append("Missing at least one digit [0-9]")
        if not any(c in _SPECIAL for c in password):
            violations.append("Missing at least one special character (e.g. !@#$%^&*)")
        return violations

    def check_policy_requirements(self, password: str) -> Dict[str, bool]:
        """
        Returns a detailed breakdown of each policy requirement.
        """
        return {
            "length_ok": len(password) >= 8,
            "has_uppercase": any(c in _UPPERCASE for c in password),
            "has_lowercase": any(c in _LOWERCASE for c in password),
            "has_digit": any(c in _DIGITS for c in password),
            "has_special": any(c in _SPECIAL for c in password),
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def load_simulator(jff_path: str) -> DFASimulator:
    """
    Loads a JFLAP .jff file and returns a ready-to-use DFASimulator.
    Raises FileNotFoundError or ValueError on parse errors.
    """
    spec = parse_jff(jff_path)
    return DFASimulator(spec, jff_path=jff_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    jff_path = sys.argv[1] if len(sys.argv) > 1 else (
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "automata", "password_dfa.jff")
    )
    sim = load_simulator(jff_path)
    print("=== DFA loaded successfully ===")
    print(describe(sim.spec))

    test_cases = [
        ("Passw0rd!", "ACCEPT – meets all requirements"),
        ("password",  "REJECT – no uppercase, no digit, no special"),
        ("SHORT1!",   "REJECT – too short (<8)"),
        ("NOLOWER1!",  "REJECT – no lowercase"),
        ("NoSpecial1", "REJECT – no special character"),
        ("12345678!A", "REJECT – no lowercase"),
    ]
    print("\n=== DFA Simulator Test Cases ===")
    for pw, expected in test_cases:
        result = sim.run(pw)
        status = "ACCEPT" if result["accepted"] else "REJECT"
        print(f"  [{status}] (expected: {expected.split('–')[0].strip()}) | "
              f"state={result['final_state']} | violations={result['violations']}")
