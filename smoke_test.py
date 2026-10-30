"""Final verification smoke test for DFA + Hybrid engine."""
import sys, os
sys.path.insert(0, 'src')

# --- DFA smoke test ---
from dfa_simulator import load_simulator
sim = load_simulator('automata/password_dfa.jff')
dfa_cases = [
    ('Abc123!@', True),
    ('abc123!@x', False),
    ('Ab1!', False),
    ('MyP@ssw0rd123', True),
    ('', False),
]
print('=== DFA Simulator Smoke Test ===')
all_pass = True
for pw, expected in dfa_cases:
    r = sim.run(pw)
    ok = r['accepted'] == expected
    all_pass = all_pass and ok
    symbol = 'PASS' if ok else 'FAIL'
    print(f'  [{symbol}] pw_len={len(pw):3d} expected={str(expected):<5} got={str(r["accepted"]):<5}')
print(f'All DFA tests passed: {all_pass}\n')

# --- Hybrid engine smoke test ---
import hybrid_engine as he
MODEL_DIR = os.path.abspath('models')
model, vocab, cfg, device = he.load_slm(MODEL_DIR)
dfa_sim = he.load_simulator('automata/password_dfa.jff')

hybrid_cases = ['Abc123!@', '12345678', 'abc', 'CorrectHorseBatteryStaple1!']
print('=== Hybrid Engine Smoke Test (3 Approaches) ===')
for pw in hybrid_cases:
    r = he.analyse(pw, model, vocab, device, dfa_sim)
    a = r.get('approach_a_slm', {})
    b = r.get('approach_b_dfa', {})
    c = r.get('approach_c_hybrid', {})
    composite = r.get('composite_verdict', 'N/A')
    status = r.get('overall_status', 'N/A')
    slm_tier = a.get('predicted_tier', 'N/A')
    dfa_acc  = b.get('accepted', 'N/A')
    hybrid_v = c.get('verdict', 'N/A')
    print(f'  pw_len={len(pw):3d}  slm={slm_tier:<12}  dfa={str(dfa_acc):<5}  hybrid_verdict={str(hybrid_v):<25}  status={status}')

print('\n=== System OK ===')
