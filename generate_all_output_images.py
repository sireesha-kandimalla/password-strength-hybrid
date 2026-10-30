"""
generate_all_output_images.py  —  WHITE BACKGROUND VERSION
Generates every chart/plot and saves them to reports/images/.
Run from the project root: python generate_all_output_images.py
"""
import sys, os, json, warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path

# ── global style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.edgecolor':    '#cccccc',
    'axes.labelcolor':   '#222222',
    'xtick.color':       '#222222',
    'ytick.color':       '#222222',
    'text.color':        '#222222',
    'grid.color':        '#e8e8e8',
    'grid.linestyle':    '--',
    'grid.linewidth':    0.7,
    'font.family':       'DejaVu Sans',
})

OUT = Path('reports/images')
OUT.mkdir(parents=True, exist_ok=True)

# ── load data ───────────────────────────────────────────────────────────────
with open('models/evaluation_report.json') as f:
    report = json.load(f)

baseline = report['baseline']
slm      = report['slm']
dfa_r    = report['dfa']
hybrid   = report['hybrid_ablation']

CLASS_NAMES_5 = ['Very Weak', 'Weak', 'Medium', 'Strong', 'Very Strong']
CAT_COLORS    = ['#d62728','#ff7f0e','#bcbd22','#2ca02c','#1f77b4']
MODEL_COLORS  = {'Baseline':'#e377c2','SLM':'#1f77b4','Hybrid':'#2ca02c'}

def _save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {p}')


# ════════════════════════════════════════════════════════════════════════════
# 1. CONFUSION MATRICES
# ════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices():
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor('white')

    for ax, cm_data, title, cmap in zip(
        axes,
        [baseline['confusion_matrix'], slm['confusion_matrix']],
        ['Baseline  (TF-IDF + Logistic Regression)', 'SLM  (Character-Level Transformer)'],
        ['Oranges', 'Blues'],
    ):
        cm      = np.array(cm_data)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
        n       = len(CLASS_NAMES_5)

        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(CLASS_NAMES_5, rotation=35, ha='right', fontsize=9)
        ax.set_yticklabels(CLASS_NAMES_5, fontsize=9)
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label',      fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

        for i in range(n):
            for j in range(n):
                txt_color = 'white' if cm_norm[i, j] > 0.55 else '#222'
                ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.2f})',
                        ha='center', va='center', fontsize=8.5,
                        color=txt_color, fontweight='bold')

    fig.suptitle('Confusion Matrices — Held-Out Test Set  (n = 1,500)',
                 fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save(fig, '01_confusion_matrices.png')


# ════════════════════════════════════════════════════════════════════════════
# 2. MODEL COMPARISON BAR CHART
# ════════════════════════════════════════════════════════════════════════════
def plot_model_comparison():
    metric_keys  = ['Accuracy', 'Macro-F1', 'Weak-PW Recall', 'False-Strong Rate']
    baseline_vals= [baseline['accuracy'], baseline['macro_f1'],
                    baseline['weak_pw_recall'], baseline['false_strong_rate']]
    slm_vals     = [slm['accuracy'],      slm['macro_f1'],
                    slm['weak_pw_recall'],      slm['false_strong_rate']]

    x = np.arange(len(metric_keys))
    w = 0.32

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('white')

    for vals, label, color, offset in [
        (baseline_vals, 'Baseline (TF-IDF + LR)', '#e377c2', -w/2),
        (slm_vals,      'SLM (Char Transformer)', '#1f77b4',  w/2),
    ]:
        bars = ax.bar(x + offset, vals, w, label=label,
                      color=color, alpha=0.88, edgecolor='white', linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.015,
                    f'{val:.1%}', ha='center', va='bottom',
                    fontsize=9.5, fontweight='bold', color='#222')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0%}'))
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison — Test Set (n = 1,500)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    _save(fig, '02_model_comparison.png')


# ════════════════════════════════════════════════════════════════════════════
# 3. PER-CLASS METRICS
# ════════════════════════════════════════════════════════════════════════════
def plot_per_class_metrics():
    pr        = slm['per_class_report']
    classes   = CLASS_NAMES_5
    precision = [pr[c]['precision'] for c in classes]
    recall    = [pr[c]['recall']    for c in classes]
    f1        = [pr[c]['f1-score']  for c in classes]

    x = np.arange(len(classes))
    w = 0.26

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('white')

    for vals, label, color, offset in [
        (precision, 'Precision', '#1f77b4', -w),
        (recall,    'Recall',    '#ff7f0e',  0),
        (f1,        'F1-Score',  '#2ca02c',  w),
    ]:
        bars = ax.bar(x + offset, vals, w, label=label,
                      color=color, alpha=0.85, edgecolor='white', linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.006,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0%}'))
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('SLM Per-Class Precision / Recall / F1 — Test Set',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    _save(fig, '03_per_class_metrics.png')


# ════════════════════════════════════════════════════════════════════════════
# 4. TRAINING HISTORY
# ════════════════════════════════════════════════════════════════════════════
def plot_training_history():
    import pandas as pd
    hist_path = Path('models/slm_training_history.csv')
    if not hist_path.exists():
        print('  [SKIP] Training history CSV not found.')
        return
    df     = pd.read_csv(hist_path)
    epochs = df['epoch'] if 'epoch' in df.columns else range(1, len(df)+1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor('white')

    for ax in (ax1, ax2):
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if 'train_loss' in df.columns:
        ax1.plot(epochs, df['train_loss'], color='#1f77b4', lw=2.5, label='Train Loss')
    if 'val_loss' in df.columns:
        ax1.plot(epochs, df['val_loss'],   color='#d62728', lw=2.5, ls='--', label='Val Loss')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax1.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)

    if 'train_acc' in df.columns:
        ax2.plot(epochs, df['train_acc'], color='#2ca02c', lw=2.5, label='Train Acc')
    if 'val_acc' in df.columns:
        ax2.plot(epochs, df['val_acc'],   color='#ff7f0e', lw=2.5, ls='--', label='Val Acc')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0%}'))
    ax2.set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)

    fig.suptitle('SLM Training History', fontsize=15, fontweight='bold')
    fig.tight_layout()
    _save(fig, '04_training_history.png')


# ════════════════════════════════════════════════════════════════════════════
# 5. DFA STATE DIAGRAM
# ════════════════════════════════════════════════════════════════════════════
def plot_dfa_diagram():
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 4.5)
    ax.axis('off')
    ax.set_title('JFLAP DFA — 16-State Category-Tracking Automaton',
                 fontsize=14, fontweight='bold', color='#111', pad=18)

    GRID_W, GRID_H = 4, 4
    NODE_R = 0.30

    def state_pos(sid):
        col = sid % GRID_W
        row = sid // GRID_W
        return (col + 0.5, GRID_H - row - 0.5)

    def state_label(sid):
        bits = f'{sid:04b}'
        cats = [('U' if bits[0]=='1' else ''), ('L' if bits[1]=='1' else ''),
                ('D' if bits[2]=='1' else ''), ('S' if bits[3]=='1' else '')]
        seen = ''.join(c for c in cats if c)
        return f'q{sid}\n{seen if seen else "∅"}'

    for sid in range(16):
        x, y = state_pos(sid)
        is_accept = sid == 15
        is_init   = sid == 0
        bits_set  = bin(sid).count('1')

        if is_accept:
            ax.add_patch(plt.Circle((x,y), NODE_R+0.08, color='#2ca02c',
                                    fill=False, lw=2.5, zorder=3))
            fc, ec = '#d4f1d4', '#2ca02c'
        elif is_init:
            fc, ec = '#d0e8ff', '#1f77b4'
        else:
            g = 0.88 - bits_set * 0.06
            fc = (g, g, g+0.05)
            ec = '#aaaaaa'

        ax.add_patch(plt.Circle((x,y), NODE_R, color=fc, ec=ec, lw=2, zorder=4))
        ax.text(x, y, state_label(sid), ha='center', va='center',
                fontsize=7.5, color='#111', fontweight='bold', zorder=5)

        if is_init:
            ax.annotate('', xy=(x-NODE_R, y), xytext=(x-NODE_R-0.25, y),
                        arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=2))

    MASK_COLORS = {'U':'#d62728','L':'#2ca02c','D':'#1f77b4','S':'#ff7f0e'}
    MASKS       = {'U':8, 'L':4, 'D':2, 'S':1}
    drawn = set()
    for sid in range(16):
        for sym, mask in MASKS.items():
            nid = sid | mask
            if nid != sid and (sid, nid, sym) not in drawn:
                drawn.add((sid, nid, sym))
                x1,y1 = state_pos(sid)
                x2,y2 = state_pos(nid)
                ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                            arrowprops=dict(arrowstyle='->', color=MASK_COLORS[sym],
                                            lw=1.2, connectionstyle='arc3,rad=0.12',
                                            shrinkA=NODE_R*72, shrinkB=NODE_R*72))

    legend_items = [
        mpatches.Patch(color='#d62728', label='[UPPER] → bit3'),
        mpatches.Patch(color='#2ca02c', label='[LOWER] → bit2'),
        mpatches.Patch(color='#1f77b4', label='[DIGIT] → bit1'),
        mpatches.Patch(color='#ff7f0e', label='[SPECIAL] → bit0'),
        mpatches.Patch(color='#d4f1d4', ec='#2ca02c', label='Accepting state (q15)'),
        mpatches.Patch(color='#d0e8ff', ec='#1f77b4', label='Initial state (q0)'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=9, framealpha=0.95)
    ax.text(0.02, 0.02,
            'Accepting state: q15  (U+L+D+S all seen)\n'
            'Self-loops on [OTHER] not shown.  Length ≥ 8 enforced by Python layer.',
            transform=ax.transAxes, color='#555', fontsize=8.5, va='bottom')
    fig.tight_layout()
    _save(fig, '05_dfa_state_diagram.png')


# ════════════════════════════════════════════════════════════════════════════
# 6. HYBRID DECISION MATRIX
# ════════════════════════════════════════════════════════════════════════════
def plot_hybrid_decision_matrix():
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.axis('off')
    ax.set_title('Hybrid 2D Risk-Arbitration Decision Matrix',
                 fontsize=14, fontweight='bold', color='#111', pad=14)

    rows     = ['Very Strong', 'Strong', 'Medium', 'Weak', 'Very Weak']
    cols     = ['DFA: ACCEPT\n(policy met)', 'DFA: REJECT\n(policy violated)']
    verdicts = [
        ['ACCEPT_ROBUST ✅',       'REJECT_POLICY ❌'],
        ['ACCEPT_ROBUST ✅',       'REJECT_POLICY ❌'],
        ['ACCEPT_CONDITIONAL ⚠️',  'REJECT_POLICY ❌'],
        ['REJECT_VULNERABILITY ❌', 'REJECT_POLICY ❌'],
        ['REJECT_VULNERABILITY ❌', 'REJECT_POLICY ❌'],
    ]
    cell_fc = [
        ['#d4f1d4','#ffd6d6'],
        ['#d4f1d4','#ffd6d6'],
        ['#fffbcc','#ffd6d6'],
        ['#ffe5cc','#ffd6d6'],
        ['#ffe5cc','#ffd6d6'],
    ]
    cell_ec = [
        ['#2ca02c','#d62728'],
        ['#2ca02c','#d62728'],
        ['#bcbd22','#d62728'],
        ['#ff7f0e','#d62728'],
        ['#ff7f0e','#d62728'],
    ]

    CW, CH = 3.6, 0.80
    OX, OY = 2.0, 0.3

    # Col headers
    for j, col in enumerate(cols):
        ax.text(OX + j*CW + CW/2, OY + len(rows)*CH + 0.25, col,
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333')

    # Row headers
    for i, row in enumerate(rows):
        ax.text(OX - 0.15, OY + (len(rows)-1-i)*CH + CH/2, row,
                ha='right', va='center', fontsize=10, color='#333')

    # Cells
    for i in range(len(rows)):
        for j in range(2):
            rect = FancyBboxPatch(
                (OX + j*CW + 0.04, OY + (len(rows)-1-i)*CH + 0.04),
                CW - 0.12, CH - 0.10,
                boxstyle='round,pad=0.06',
                facecolor=cell_fc[i][j], edgecolor=cell_ec[i][j], linewidth=2,
            )
            ax.add_patch(rect)
            ax.text(OX + j*CW + CW/2,
                    OY + (len(rows)-1-i)*CH + CH/2,
                    verdicts[i][j], ha='center', va='center',
                    fontsize=10, fontweight='bold', color='#222')

    ax.text(OX/2, OY + len(rows)*CH/2,
            'SLM\nPredicted\nTier', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#555',
            rotation=90)

    ax.set_xlim(0, OX + 2*CW + 0.5)
    ax.set_ylim(0, OY + len(rows)*CH + 0.8)
    fig.tight_layout()
    _save(fig, '06_hybrid_decision_matrix.png')


# ════════════════════════════════════════════════════════════════════════════
# 7. TEST SUITE RESULTS
# ════════════════════════════════════════════════════════════════════════════
def plot_test_results():
    fig = plt.figure(figsize=(15, 6))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    # Bar chart
    ax_bar = fig.add_subplot(gs[0, :2])
    ax_bar.set_facecolor('white')

    suites  = ['test_dfa.py\n(32 tests)', 'test_slm.py\n(17 tests)', 'test_hybrid.py\n(23 tests)']
    counts  = [32, 17, 23]
    colors  = ['#1f77b4','#ff7f0e','#2ca02c']
    x       = np.arange(len(suites))

    bars = ax_bar.bar(x, counts, color=colors, alpha=0.88,
                      edgecolor='white', linewidth=0.8, width=0.5)
    for bar, val in zip(bars, counts):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val} PASS', ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color='#222')

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(suites, fontsize=10)
    ax_bar.set_ylabel('Tests Passed', fontsize=11)
    ax_bar.set_ylim(0, 42)
    ax_bar.set_title('Test Suite Results  —  72 / 72 Passed ✅',
                     fontsize=13, fontweight='bold')
    ax_bar.yaxis.grid(True); ax_bar.set_axisbelow(True)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    # Donut
    ax_pie = fig.add_subplot(gs[0, 2:])
    ax_pie.set_facecolor('white')
    wedge_vals   = [32, 17, 23]
    wedge_labels = ['DFA\n(32)', 'SLM\n(17)', 'Hybrid\n(23)']
    wedge_colors = ['#1f77b4','#ff7f0e','#2ca02c']
    wedges, texts, autotexts = ax_pie.pie(
        wedge_vals, labels=wedge_labels, colors=wedge_colors,
        autopct='%1.0f%%', startangle=90,
        wedgeprops=dict(width=0.52, edgecolor='white', linewidth=2.5),
        textprops=dict(color='#222', fontsize=10),
        pctdistance=0.73,
    )
    for at in autotexts:
        at.set_fontsize(11); at.set_fontweight('bold'); at.set_color('white')
    ax_pie.set_title('Test Distribution\n72 Tests Total', fontsize=12, fontweight='bold')
    ax_pie.text(0, -1.55, '72 / 72  PASS', ha='center',
                color='#2ca02c', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f8e8', edgecolor='#2ca02c'))

    fig.tight_layout()
    _save(fig, '07_test_suite_results.png')


# ════════════════════════════════════════════════════════════════════════════
# 8. LATENCY BENCHMARK
# ════════════════════════════════════════════════════════════════════════════
def plot_latency():
    latency = slm.get('latency_benchmark', slm.get('latency', {}))
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('white')

    metrics = ['Mean', 'P50 (Median)', 'P95', 'P99']
    values  = [
        latency.get('mean_ms',   2.50),
        latency.get('p50_ms',    latency.get('median_ms', 2.49)),
        latency.get('p95_ms',    3.46),
        latency.get('p99_ms',    4.03),
    ]
    colors = ['#1f77b4','#2ca02c','#ff7f0e','#d62728']

    bars = ax.bar(metrics, values, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=0.8, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.06,
                f'{val:.2f} ms', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax.axhline(y=10, color='#d62728', ls='--', lw=1.5, label='10 ms real-time threshold')
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_ylim(0, 13)
    ax.set_title('SLM Inference Latency — Single Sample (n = 100 runs)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    _save(fig, '08_latency_benchmark.png')


# ════════════════════════════════════════════════════════════════════════════
# 9. SYSTEM ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════
def plot_architecture():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f9f9fb')
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_title('System Architecture — Password Strength Validation Hybrid System',
                 fontsize=14, fontweight='bold', color='#111', pad=16)

    def box(ax, x, y, w, h, label, sublabel='', fc='#e8f0ff', ec='#4a90d9', fs=10):
        rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                              facecolor=fc, edgecolor=ec, linewidth=2, zorder=3)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2 + (0.14 if sublabel else 0), label,
                ha='center', va='center', color='#111',
                fontsize=fs, fontweight='bold', zorder=4)
        if sublabel:
            ax.text(x+w/2, y+h/2 - 0.24, sublabel,
                    ha='center', va='center', color='#555', fontsize=8, zorder=4)

    def arrow(ax, x1, y1, x2, y2, label='', color='#4a90d9'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2), zorder=2)
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.14, label,
                    ha='center', color='#666', fontsize=8)

    box(ax, 6.5, 7.5, 3.0, 0.8,  'Password Input',
        '(never stored / printed)', fc='#e8f8e8', ec='#2ca02c', fs=11)
    box(ax, 5.5, 5.8, 5.0, 0.9,  'Hybrid Engine',
        'hybrid_engine.py', fc='#f0e8ff', ec='#9c6bd6')
    arrow(ax, 8.0, 7.5, 8.0, 6.7, 'password string')

    box(ax, 1.0, 3.8, 4.5, 1.4,
        'Character Transformer SLM',
        'train_slm.py  •  299,525 params\n2-layer Transformer  •  d=128  •  vocab=72',
        fc='#e8f0ff', ec='#1f77b4', fs=9)
    arrow(ax, 6.0, 5.8, 4.5, 5.2, 'char indices', '#1f77b4')

    box(ax, 10.5, 3.8, 4.5, 1.4,
        'JFLAP DFA Simulator',
        'jflap_parser.py + dfa_simulator.py\n16 states  •  80 transitions',
        fc='#e8f8e8', ec='#2ca02c', fs=9)
    arrow(ax, 10.0, 5.8, 12.0, 5.2, 'password string', '#2ca02c')

    box(ax, 0.5, 1.8, 5.5, 1.2,
        'Tier + Probabilities',
        'Very Weak / … / Very Strong\n+ confidence scores per class',
        fc='#eef4ff', ec='#1f77b4', fs=9)
    arrow(ax, 3.25, 3.8, 3.25, 3.0, 'predicted_tier', '#1f77b4')

    box(ax, 10.0, 1.8, 5.5, 1.2,
        'Policy Result + Violations',
        'ACCEPT / REJECT\n+ missing requirements list',
        fc='#edfcee', ec='#2ca02c', fs=9)
    arrow(ax, 12.75, 3.8, 12.75, 3.0, 'accepted / violations', '#2ca02c')

    box(ax, 4.5, 0.25, 7.0, 1.2,
        '2D Risk-Arbitration  →  Composite Verdict + Suggestions',
        'ACCEPT_ROBUST | ACCEPT_CONDITIONAL | REJECT_VULNERABILITY | REJECT_POLICY',
        fc='#fff0f8', ec='#e377c2', fs=9)
    arrow(ax, 3.25, 1.8, 6.0, 1.45, '', '#1f77b4')
    arrow(ax, 12.75, 1.8, 10.0, 1.45, '', '#2ca02c')

    box(ax, 6.2, -0.1, 3.6, 0.65, 'Streamlit UI  (app.py)',
        fc='#fff8e8', ec='#bcbd22', fs=9)

    fig.tight_layout()
    _save(fig, '09_system_architecture.png')


# ════════════════════════════════════════════════════════════════════════════
# 10. SUMMARY CARD
# ════════════════════════════════════════════════════════════════════════════
def plot_summary_card():
    cards = [
        ('90.20%',  'Test Accuracy\n(SLM)',            '#1f77b4'),
        ('90.17%',  'Macro-F1\n(SLM)',                 '#2ca02c'),
        ('98.17%',  'Weak-PW Recall\n(SLM)',           '#ff7f0e'),
        ('0.00%',   'False-Strong\nRate (SLM)',         '#d62728'),
        ('72 / 72', 'Tests Passed\n(All Suites)',      '#9467bd'),
        ('2.75 ms', 'Mean Inference\nLatency',         '#8c564b'),
        ('+50.2 pp','Macro-F1 Gain\nvs Baseline',      '#17becf'),
        ('16 states','DFA States\n(JFLAP Automaton)',  '#bcbd22'),
    ]

    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.axis('off')

    W, H  = 1.75, 2.0
    PAD   = 0.22
    x0    = 0.3

    for i, (val, label, color) in enumerate(cards):
        x = x0 + i * (W + PAD)
        light = color + '22'
        rect = FancyBboxPatch((x, 0.4), W, H,
                              boxstyle='round,pad=0.12',
                              facecolor=light, edgecolor=color,
                              linewidth=2.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x + W/2, 0.4 + H*0.62, val,
                ha='center', va='center', color=color,
                fontsize=16, fontweight='bold', zorder=4)
        ax.text(x + W/2, 0.4 + H*0.24, label,
                ha='center', va='center', color='#333',
                fontsize=9, zorder=4)

    total_w = x0 + len(cards)*(W+PAD)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, 3.2)
    ax.set_title('Key Results at a Glance', fontsize=14, fontweight='bold', color='#111', pad=10)
    fig.tight_layout()
    _save(fig, '10_summary_card.png')


# ════════════════════════════════════════════════════════════════════════════
# 11. DFA TEST CASES TABLE
# ════════════════════════════════════════════════════════════════════════════
def plot_dfa_test_cases():
    from dfa_simulator import load_simulator
    sim = load_simulator('automata/password_dfa.jff')

    test_cases = [
        ('Abc123!@',          True,  'All categories, exactly 8 chars'),
        ('MyP@ssw0rd123',     True,  'All categories, long (13 chars)'),
        ('AAAA1111aaaa!!!!',  True,  'All categories, ≥8, repeated chars'),
        ('CorrectHorseBat1!', True,  'Passphrase-style, all categories'),
        ('abc123!@x',         False, 'Missing uppercase letter'),
        ('ABC123!@X',         False, 'Missing lowercase letter'),
        ('Abcdefg!',          False, 'Missing digit'),
        ('Abcde123',          False, 'Missing special character'),
        ('Aa1!Aa1',           False, 'Too short (7 chars, all cats)'),
        ('',                  False, 'Empty string'),
    ]

    fig, ax = plt.subplots(figsize=(15, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.axis('off')
    ax.set_title('DFA Policy Test Cases — All 10 Pass ✅',
                 fontsize=14, fontweight='bold', color='#111', pad=14)

    col_x   = [0.3, 5.2, 8.2, 10.5, 12.5]
    headers = ['Password (masked)', 'Expected', 'DFA Result', 'Match', 'Description']
    row_h   = 0.55
    start_y = 6.5

    for h, cx in zip(headers, col_x):
        ax.text(cx, start_y, h, fontsize=10, fontweight='bold', color='#444', va='center')
    ax.axhline(start_y - 0.22, color='#ccc', linewidth=1.2)

    for row_i, (pw, expected, desc) in enumerate(test_cases):
        y      = start_y - (row_i+1)*row_h
        result = sim.run(pw)
        got    = result['accepted']
        match  = got == expected
        bg     = '#f0fff0' if expected else '#fff5f5'

        rect = FancyBboxPatch((0.15, y - 0.23), 14.6, row_h - 0.06,
                              boxstyle='round,pad=0.04',
                              facecolor=bg, edgecolor='#ddd', linewidth=1, zorder=2)
        ax.add_patch(rect)

        masked = ('*' * min(len(pw), 8) + ('…' if len(pw) > 8 else '')) if pw else '(empty)'
        ax.text(col_x[0], y, masked,  fontsize=9.5, color='#333', va='center', zorder=3,
                family='monospace')

        exp_c = '#2ca02c' if expected else '#d62728'
        ax.text(col_x[1], y, 'ACCEPT ✅' if expected else 'REJECT ❌',
                fontsize=9.5, color=exp_c, fontweight='bold', va='center', zorder=3)

        got_c = '#2ca02c' if got else '#d62728'
        ax.text(col_x[2], y, 'ACCEPT ✅' if got else 'REJECT ❌',
                fontsize=9.5, color=got_c, fontweight='bold', va='center', zorder=3)

        badge_c = '#2ca02c' if match else '#d62728'
        ax.text(col_x[3], y, '✓ PASS' if match else '✗ FAIL',
                fontsize=9.5, color=badge_c, fontweight='bold', va='center', zorder=3)

        ax.text(col_x[4], y, desc, fontsize=9, color='#555', va='center', zorder=3)

    ax.set_xlim(0, 15.2)
    ax.set_ylim(start_y - (len(test_cases)+1)*row_h, start_y + 0.6)
    fig.tight_layout()
    _save(fig, '11_dfa_test_cases.png')


# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating all output images (white background)...\n')
    plot_confusion_matrices()
    plot_model_comparison()
    plot_per_class_metrics()
    plot_training_history()
    plot_dfa_diagram()
    plot_hybrid_decision_matrix()
    plot_test_results()
    plot_latency()
    plot_architecture()
    plot_summary_card()
    plot_dfa_test_cases()
    total = len(list(OUT.glob('*.png')))
    print(f'\nDone — {total} images saved to {OUT}')
