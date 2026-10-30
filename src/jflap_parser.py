"""
jflap_parser.py
---------------
Parser and specification classes for JFLAP .jff finite automaton files.

Supports:
- Parsing standard JFLAP 7.1 XML (.jff) files for Finite Automata (type="fa").
- Category-label transitions: [UPPER], [LOWER], [DIGIT], [SPECIAL], [OTHER]
  as well as single-character symbols: 'U', 'L', 'D', 'S', 'X'.
- DFASpec dataclass matching both native simulator and extended analysis.
- DOT export and automaton consistency validation.
- Zero-exposure: passwords are never printed or logged.
"""

from __future__ import annotations

import os
import string
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Constants & Symbol Mappings
# ---------------------------------------------------------------------------

SPECIAL_CHARS: frozenset[str] = frozenset(
    r"""!@#$%^&*()_+-=[]{}|;':",./<>?`~\\"""
)

LABEL_TO_SYMBOL: dict[str, str] = {
    "[UPPER]": "U",
    "[LOWER]": "L",
    "[DIGIT]": "D",
    "[SPECIAL]": "S",
    "[OTHER]": "X",
}

SYMBOL_TO_LABEL: dict[str, str] = {
    "U": "[UPPER]",
    "L": "[LOWER]",
    "D": "[DIGIT]",
    "S": "[SPECIAL]",
    "X": "[OTHER]",
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class StateInfo:
    id: int
    name: str
    is_initial: bool = False
    is_final: bool = False
    x: float = 0.0
    y: float = 0.0


@dataclass
class TransitionInfo:
    from_state: int
    to_state: int
    read: str


@dataclass
class DFASpec:
    """
    Representation of parsed DFA structure consumed by DFASimulator.
    - states: mapping of state id (str) to state name (str)
    - initial_state: initial state id (str)
    - accepting_states: set of accepting state ids (set[str])
    - transitions: mapping of (from_state_id, symbol) -> to_state_id (all str)
    - alphabet: set of input symbols recognized by transitions
    """
    states: Dict[str, str]
    initial_state: str
    accepting_states: Set[str]
    transitions: Dict[Tuple[str, str], str]
    alphabet: Set[str]


# ---------------------------------------------------------------------------
# parse_jff & describe functions
# ---------------------------------------------------------------------------

def parse_jff(filepath: Union[str, Path]) -> DFASpec:
    """
    Parses a JFLAP .jff XML file and returns a DFASpec instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If XML is malformed or not a valid JFLAP FA.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"JFLAP file not found: {path}")

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Malformed XML in JFLAP file {path}: {e}")

    automaton = root.find("automaton")
    if automaton is None:
        raise ValueError(f"No <automaton> element found in {path}")

    states: Dict[str, str] = {}
    initial_state: Optional[str] = None
    accepting_states: Set[str] = set()
    transitions: Dict[Tuple[str, str], str] = {}
    alphabet: Set[str] = set()

    for state_elem in automaton.findall("state"):
        s_id = state_elem.get("id")
        s_name = state_elem.get("name", f"q{s_id}")
        if s_id is None:
            continue
        states[s_id] = s_name

        if state_elem.find("initial") is not None:
            if initial_state is not None:
                raise ValueError(f"Multiple initial states detected in {path}")
            initial_state = s_id

        if state_elem.find("final") is not None:
            accepting_states.add(s_id)

    if initial_state is None:
        raise ValueError(f"DFA has no initial state defined in {path}")

    for trans_elem in automaton.findall("transition"):
        from_el = trans_elem.find("from")
        to_el = trans_elem.find("to")
        read_el = trans_elem.find("read")

        if from_el is None or to_el is None:
            continue

        from_id = from_el.text.strip() if from_el.text else ""
        to_id = to_el.text.strip() if to_el.text else ""
        read_sym = read_el.text.strip() if (read_el is not None and read_el.text) else ""

        # Normalize bracketed labels to single characters if necessary
        canonical_sym = LABEL_TO_SYMBOL.get(read_sym, read_sym)
        transitions[(from_id, canonical_sym)] = to_id
        alphabet.add(canonical_sym)

    return DFASpec(
        states=states,
        initial_state=initial_state,
        accepting_states=accepting_states,
        transitions=transitions,
        alphabet=alphabet,
    )


def describe(spec: DFASpec) -> str:
    """Returns a formatted human-readable summary of the DFA specification."""
    init_name = spec.states.get(spec.initial_state, spec.initial_state)
    acc_names = [spec.states.get(s, s) for s in sorted(spec.accepting_states)]
    lines = [
        "JFLAP DFA Specification Summary:",
        f"  Total States     : {len(spec.states)}",
        f"  Initial State    : {init_name} (id={spec.initial_state})",
        f"  Accepting States : {len(spec.accepting_states)} ({', '.join(acc_names)})",
        f"  Alphabet ({len(spec.alphabet)})   : {sorted(spec.alphabet)}",
        f"  Transitions      : {len(spec.transitions)} total defined transitions",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JFLAPParser Class (Object-Oriented Interface & DOT Generation)
# ---------------------------------------------------------------------------

class JFLAPParser:
    """Parse a JFLAP .jff file and return detailed DFA structure."""

    def __init__(self, jff_path: Union[str, Path]) -> None:
        self.jff_path = Path(jff_path)
        self._parsed: Optional[dict] = None

    def parse(self) -> dict:
        if self._parsed is not None:
            return self._parsed

        spec = parse_jff(self.jff_path)

        state_objects: dict[int, StateInfo] = {}
        for sid_str, sname in spec.states.items():
            sid = int(sid_str)
            state_objects[sid] = StateInfo(
                id=sid,
                name=sname,
                is_initial=(sid_str == spec.initial_state),
                is_final=(sid_str in spec.accepting_states),
            )

        trans_list: list[TransitionInfo] = []
        for (from_str, sym), to_str in spec.transitions.items():
            trans_list.append(TransitionInfo(
                from_state=int(from_str),
                to_state=int(to_str),
                read=sym,
            ))

        self._parsed = {
            "states": state_objects,
            "initial_state": int(spec.initial_state),
            "accepting_states": {int(s) for s in spec.accepting_states},
            "transitions": trans_list,
            "alphabet": spec.alphabet,
            "spec": spec,
        }
        return self._parsed

    def to_dot(self) -> str:
        """Export the DFA as a Graphviz DOT string."""
        data = self.parse()
        states: dict[int, StateInfo] = data["states"]
        transitions: list[TransitionInfo] = data["transitions"]
        accepting: set[int] = data["accepting_states"]
        initial: int = data["initial_state"]

        lines: list[str] = ["digraph DFA {", "  rankdir=LR;", "  node [shape=circle];"]
        for sid, s in states.items():
            shape = "doublecircle" if sid in accepting else "circle"
            lines.append(f'  {sid} [label="{s.name}", shape={shape}];')

        lines.append('  __start [shape=none, label=""];')
        lines.append(f"  __start -> {initial};")

        edge_labels: dict[tuple[int, int], list[str]] = {}
        for t in transitions:
            key = (t.from_state, t.to_state)
            edge_labels.setdefault(key, []).append(t.read)

        for (frm, to), labels in edge_labels.items():
            label = ", ".join(labels)
            lines.append(f'  {frm} -> {to} [label="{label}"];')

        lines.append("}")
        return "\n".join(lines)

    def validate_automaton(self) -> list[str]:
        data = self.parse()
        states = data["states"]
        transitions = data["transitions"]
        accepting = data["accepting_states"]
        initial = data["initial_state"]
        alphabet = data["alphabet"]

        warnings: list[str] = []
        trans_map: dict[tuple[int, str], int] = {}
        for t in transitions:
            key = (t.from_state, t.read)
            if key in trans_map:
                warnings.append(
                    f"Non-determinism: state {states[t.from_state].name!r} has multiple transitions on {t.read!r}."
                )
            trans_map[key] = t.to_state

        for sid, s in states.items():
            for sym in alphabet:
                if (sid, sym) not in trans_map:
                    warnings.append(
                        f"Missing transition: state {s.name!r} has no transition on {sym!r}."
                    )

        reachable: set[int] = {initial}
        queue: list[int] = [initial]
        while queue:
            cur = queue.pop(0)
            for sym in alphabet:
                nxt = trans_map.get((cur, sym))
                if nxt is not None and nxt not in reachable:
                    reachable.add(nxt)
                    queue.append(nxt)

        for sid, s in states.items():
            if sid not in reachable:
                warnings.append(f"Unreachable state: {s.name!r} (id={sid}).")

        if not accepting:
            warnings.append("No accepting states defined.")

        return warnings


# ---------------------------------------------------------------------------
# PasswordDFA and Helper Functions
# ---------------------------------------------------------------------------

class PasswordDFA:
    """Alternative convenience wrapper for category tracking DFA."""

    def __init__(self, automaton: dict) -> None:
        self._states = automaton["states"]
        self._initial = automaton["initial_state"]
        self._accepting = automaton["accepting_states"]
        self._transitions = automaton["transitions"]
        self._trans_map = {(t.from_state, t.read): t.to_state for t in self._transitions}

    @staticmethod
    def _char_to_category(ch: str) -> str:
        if ch in string.ascii_uppercase:
            return "U"
        if ch in string.ascii_lowercase:
            return "L"
        if ch in string.digits:
            return "D"
        if ch in SPECIAL_CHARS:
            return "S"
        return "X"

    def run(self, password: str) -> dict:
        length_ok = len(password) >= 8
        current = self._initial
        path = []
        categories_seen = set()

        for ch in password:
            cat = self._char_to_category(ch)
            categories_seen.add(cat)
            next_state = self._trans_map.get((current, cat), current)
            path.append((self._states[current].name, cat, self._states[next_state].name))
            current = next_state

        accepted = current in self._accepting and length_ok
        missing_categories = []
        if "U" not in categories_seen:
            missing_categories.append("uppercase letter (A-Z)")
        if "L" not in categories_seen:
            missing_categories.append("lowercase letter (a-z)")
        if "D" not in categories_seen:
            missing_categories.append("digit (0-9)")
        if "S" not in categories_seen:
            missing_categories.append("special character (!@#$...)")
        if not length_ok:
            missing_categories.insert(0, f"minimum length of 8 characters (got {len(password)})")

        return {
            "accepted": accepted,
            "length_ok": length_ok,
            "policy_met": accepted,
            "final_state": self._states[current].name,
            "categories_seen": categories_seen,
            "missing_categories": missing_categories,
            "path": path,
            "error": None,
        }


def load_dfa(jff_path: Union[str, Path]) -> PasswordDFA:
    parser = JFLAPParser(jff_path)
    automaton = parser.parse()
    return PasswordDFA(automaton)


def generate_test_cases() -> list[dict]:
    return [
        {"password": "Abc123!@", "expected_policy_met": True, "description": "8 chars — upper, lower, digit, special"},
        {"password": "MyP@ssw0rd123", "expected_policy_met": True, "description": "13 chars with all four categories"},
        {"password": "abc123!@", "expected_policy_met": False, "description": "8 chars but no uppercase letter"},
        {"password": "ABC123!@", "expected_policy_met": False, "description": "8 chars but no lowercase letter"},
        {"password": "Abcdefg!", "expected_policy_met": False, "description": "8 chars but no digit"},
        {"password": "Abcde123", "expected_policy_met": False, "description": "8 chars but no special character"},
        {"password": "Ab1!", "expected_policy_met": False, "description": "All 4 categories present but only 4 chars (too short)"},
        {"password": "", "expected_policy_met": False, "description": "Empty string"},
        {"password": "AAAA1111aaaa!!!!", "expected_policy_met": True, "description": "16 chars with all categories"},
        {"password": "Aa1!Aa1", "expected_policy_met": False, "description": "7 chars (one short)"},
    ]
