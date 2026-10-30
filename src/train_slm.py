"""
train_slm.py
------------
Trains the Small Language Model (SLM): a character-level Transformer classifier
for password strength prediction (5 classes).
Architecture: Char embedding → Positional encoding → 2-layer Transformer encoder
              → masked mean pooling → classification head.
Saves model, vocabulary, config, and training history to models/.
"""

import os
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

NUM_CLASSES = 5
PAD_IDX = 0
UNK_IDX = 1
MAX_LEN = 64
BATCH_SIZE = 128
NUM_EPOCHS = 30
EARLY_STOP_PATIENCE = 5
LEARNING_RATE = 3e-4
D_MODEL = 128
NHEADS = 4
NUM_LAYERS = 2
DFF = 256
DROPOUT = 0.2

CLASS_NAMES = ["Very Weak", "Weak", "Medium", "Strong", "Very Strong"]


# ── Vocabulary Builder ───────────────────────────────────────────────────────

def build_vocabulary(passwords, min_freq: int = 1):
    from collections import Counter
    counts = Counter()
    for pw in passwords:
        counts.update(str(pw))
    vocab = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    for char, freq in sorted(counts.items()):
        if freq >= min_freq and char not in vocab:
            vocab[char] = len(vocab)
    return vocab


# ── Dataset ──────────────────────────────────────────────────────────────────

class PasswordDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: dict, max_len: int = MAX_LEN):
        self.passwords = df["password"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.passwords)

    def encode(self, pw: str):
        ids = [self.vocab.get(c, UNK_IDX) for c in pw[: self.max_len]]
        pad_len = self.max_len - len(ids)
        return ids + [PAD_IDX] * pad_len

    def __getitem__(self, idx):
        ids = self.encode(self.passwords[idx])
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def collate_fn(batch):
    tokens, labels = zip(*batch)
    return torch.stack(tokens), torch.stack(labels)


# ── Model ────────────────────────────────────────────────────────────────────

class CharTransformerSLM(nn.Module):
    """
    Compact character-level Transformer encoder for password strength classification.
    Design: embedding → sinusoidal pos-enc → TransformerEncoder (2 layers, 4 heads)
            → masked mean pooling → MLP head.
    """

    def __init__(self, vocab_size, num_classes, d_model, nhead, num_layers,
                 dim_feedforward, dropout, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.max_len = max_len

    def forward(self, token_ids: torch.Tensor):
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(token_ids) + self.pos_enc(positions)
        padding_mask = (token_ids == PAD_IDX)
        all_padded = padding_mask.all(dim=-1)
        if all_padded.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_padded, 0] = False
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        # Masked mean pooling (ignore PAD positions)
        mask = (~padding_mask).unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for tokens, labels in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(tokens)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for tokens, labels in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        logits = model(tokens)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def train_slm(train_csv: str, val_csv: str, model_dir: str):
    os.makedirs(model_dir, exist_ok=True)

    print("Loading data...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    print("Building vocabulary from training set...")
    vocab = build_vocabulary(train_df["password"])
    vocab_path = os.path.join(model_dir, "char_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    print(f"  Vocabulary size: {len(vocab)} tokens")

    # Class weights from training set (inverse frequency)
    label_counts = train_df["label"].value_counts().sort_index()
    weights = 1.0 / label_counts.values
    weights = weights / weights.sum() * NUM_CLASSES
    class_weights = torch.tensor(weights, dtype=torch.float32)

    train_ds = PasswordDataset(train_df, vocab)
    val_ds = PasswordDataset(val_df, vocab)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    model = CharTransformerSLM(
        vocab_size=len(vocab),
        num_classes=NUM_CLASSES,
        d_model=D_MODEL,
        nhead=NHEADS,
        num_layers=NUM_LAYERS,
        dim_feedforward=DFF,
        dropout=DROPOUT,
        max_len=MAX_LEN,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = []
    best_val_f1 = -1.0
    best_epoch = 0
    patience_ctr = 0
    best_model_path = os.path.join(model_dir, "slm_best.pt")

    print("\nTraining SLM...")
    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Compute macro-F1 on val set
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for tokens, labels in val_loader:
                logits = model(tokens.to(device))
                all_preds.extend(logits.argmax(1).cpu().tolist())
                all_labels.extend(labels.tolist())
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        row = dict(epoch=epoch, train_loss=t_loss, train_acc=t_acc,
                   val_loss=v_loss, val_acc=v_acc, val_macro_f1=val_f1)
        history.append(row)
        print(f"  Epoch {epoch:02d} | TrainLoss={t_loss:.4f} Acc={t_acc:.4f} | "
              f"ValLoss={v_loss:.4f} Acc={v_acc:.4f} F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best F1={best_val_f1:.4f} @ epoch {best_epoch})")
                break

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f}s")
    print(f"Best Val Macro-F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Save config and history
    config = {
        "vocab_size": len(vocab),
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "d_model": D_MODEL,
        "nhead": NHEADS,
        "num_layers": NUM_LAYERS,
        "dim_feedforward": DFF,
        "dropout": DROPOUT,
        "max_len": MAX_LEN,
        "num_params": num_params,
        "best_val_macro_f1": best_val_f1,
        "best_epoch": best_epoch,
        "training_time_seconds": round(training_time, 1),
    }
    with open(os.path.join(model_dir, "slm_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(model_dir, "slm_training_history.csv"), index=False)

    return config


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "processed", "train.csv")
    val_path = os.path.join(base_dir, "data", "processed", "val.csv")
    model_dir = os.path.join(base_dir, "models")
    train_slm(train_path, val_path, model_dir)
