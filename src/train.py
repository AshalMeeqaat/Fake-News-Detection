"""
=============================================================================
ASSIGNMENT 3 — EXPERIMENTATION AND EXPANSION
"DistilBERT with Focal Loss for Robust Fake News Detection"
Muhammad Saif Murtaza, Eeshaal Adeel, Ash Al Meeqaat | CS-4112 | FAST-NUCES
=============================================================================
"""

import os, warnings, re, time
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]    = "3"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim      import AdamW
from transformers     import (AutoTokenizer,
                               AutoModelForSequenceClassification,
                               get_cosine_schedule_with_warmup)
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (accuracy_score, precision_score,
                                      recall_score, f1_score,
                                      classification_report,
                                      confusion_matrix, ConfusionMatrixDisplay)
import nltk
nltk.download("stopwords", quiet=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[v] Device : {device}")
if device.type == "cuda":
    print(f"[v] GPU    : {torch.cuda.get_device_name(0)}")
    print(f"[v] VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# -- Config -------------------------------------------------------------------
MODEL_NAME      = "distilbert-base-uncased"   # 66M params, 3x faster than BERT
MAX_LEN         = 128
BATCH_SIZE      = 32
EPOCHS          = 3
LR              = 2e-5
TRAIN_SUBSAMPLE = None
FOCAL_GAMMA     = 2.0
FOCAL_ALPHA     = 0.25

BASE_SVM_ACC = 0.9903
BASE_MLP_ACC = 0.9877


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs      = F.softmax(logits, dim=1)
        targets_1h = F.one_hot(targets, num_classes=2).float()
        pt         = (probs * targets_1h).sum(dim=1)
        focal_w    = self.alpha * (1 - pt) ** self.gamma
        ce_loss    = F.cross_entropy(logits, targets, reduction="none")
        return (focal_w * ce_loss).mean()


def tokenize_all(texts, labels, tokenizer, max_len):
    enc = tokenizer(list(texts), max_length=max_len, padding="max_length",
                    truncation=True, return_tensors="pt")
    return TensorDataset(enc["input_ids"], enc["attention_mask"],
                         torch.tensor(labels.astype(int), dtype=torch.long))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+|#\w+",      " ", text)
    text = re.sub(r"[^a-z\s]",       " ", text)
    return re.sub(r"\s+", " ", text).strip()

def get_metrics(y_true, y_pred, label=""):
    acc  = accuracy_score (y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score   (y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score       (y_true, y_pred, average="weighted", zero_division=0)
    print(f"  {label:<55}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    return dict(model=label, accuracy=acc, precision=prec, recall=rec, f1=f1)

def evaluate_loader(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for ids, mask, lbls in loader:
            ids, mask = ids.to(device), mask.to(device)
            out = model(input_ids=ids, attention_mask=mask)
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            trues.extend(lbls.numpy())
    return np.array(trues), np.array(preds)


# =============================================================================
print("\n" + "="*70)
print("STEP 1: LOADING PRIMARY DATASET -- TruthSeeker2023")
print("="*70)

TS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Truth_Seeker_Model_Dataset.csv")
if not os.path.exists(TS_PATH):
    raise FileNotFoundError(f"[!] '{TS_PATH}' not found.")

df_ts = pd.read_csv(TS_PATH)
TEXT_COL  = next((c for c in ["tweet","text","content","body"]
                  if c in df_ts.columns), df_ts.columns[0])
LABEL_COL = "BinaryNumTarget" if "BinaryNumTarget" in df_ts.columns else df_ts.columns[-1]

df_ts          = df_ts[[TEXT_COL, LABEL_COL]].dropna()
df_ts.columns  = ["text", "label"]
df_ts["label"] = df_ts["label"].astype(int).astype("int64")
df_ts["clean"] = df_ts["text"].apply(clean_text)
df_ts          = df_ts[df_ts["clean"].str.len() > 0]
print(f"TruthSeeker: {len(df_ts)} samples | Labels: {dict(df_ts['label'].value_counts())}")

X_ts = df_ts["clean"].to_numpy(dtype=str)
y_ts = df_ts["label"].to_numpy(dtype=int)

X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_ts, y_ts, test_size=0.30,
                                              stratify=y_ts, random_state=SEED)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50,
                                              stratify=y_tmp, random_state=SEED)
print(f"Split -- Train={len(y_tr)}  Val={len(y_val)}  Test={len(y_te)}")

if TRAIN_SUBSAMPLE and len(X_tr) > TRAIN_SUBSAMPLE:
    X_tr_mb, _, y_tr_mb, _ = train_test_split(X_tr, y_tr, train_size=TRAIN_SUBSAMPLE,
                                               stratify=y_tr, random_state=SEED)
    print(f"Training subsample: {TRAIN_SUBSAMPLE} (stratified)")
else:
    X_tr_mb, y_tr_mb = X_tr, y_tr


print("\n" + "="*70)
print("STEP 2: LOADING CROSS-DOMAIN DATASET -- ISOT")
print("="*70)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ISOT_AVAILABLE = os.path.exists(os.path.join(DATA_DIR, "True.csv")) and os.path.exists(os.path.join(DATA_DIR, "Fake.csv"))
if ISOT_AVAILABLE:
    df_true         = pd.read_csv(os.path.join(DATA_DIR, "True.csv"));  df_true["label"] = 1
    df_fake         = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"));  df_fake["label"] = 0
    df_isot         = pd.concat([df_true, df_fake], ignore_index=True)
    text_col        = "text" if "text" in df_isot.columns else df_isot.columns[0]
    df_isot         = df_isot[[text_col, "label"]].dropna()
    df_isot.columns = ["text", "label"]
    df_isot["clean"]= df_isot["text"].apply(clean_text)
    df_isot         = df_isot[df_isot["clean"].str.len() > 0]
    df_isot         = df_isot.sample(n=min(20000, len(df_isot)), random_state=SEED)
    print(f"ISOT: {len(df_isot)} samples | Labels: {dict(df_isot['label'].value_counts())}")
    X_isot = df_isot["clean"].to_numpy(dtype=str)
    y_isot = df_isot["label"].to_numpy(dtype=int)
else:
    print("[!] ISOT not found. Cross-domain eval will be SKIPPED.")
    X_isot = y_isot = None


print("\n" + "="*70)
print("STEP 3: DISTILBERT + FOCAL LOSS (PROPOSED METHOD)")
print("="*70)
print(f"Model: {MODEL_NAME} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")
print(f"Train N: {len(y_tr_mb)}")

print("Loading tokenizer ...", end=" ", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("done")

print("Pre-tokenizing all splits (one-time) ...")
t0 = time.time()
train_ds = tokenize_all(X_tr_mb, y_tr_mb, tokenizer, MAX_LEN)
val_ds   = tokenize_all(X_val,   y_val,   tokenizer, MAX_LEN)
test_ds  = tokenize_all(X_te,    y_te,    tokenizer, MAX_LEN)
print(f"Done in {time.time()-t0:.1f}s")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=128,        shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=128,        shuffle=False, num_workers=0, pin_memory=True)

print("Loading model ...", end=" ", flush=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)
model.to(device)
print(f"done ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

focal_loss_fn = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
total_steps   = len(train_loader) * EPOCHS
scheduler     = get_cosine_schedule_with_warmup(optimizer,
                    num_warmup_steps=int(0.06*total_steps),
                    num_training_steps=total_steps)

history      = {"train_loss": [], "val_acc": [], "val_f1": []}
best_val_acc = 0
best_epoch   = 0

print("\nTraining ...")
for epoch in range(EPOCHS):
    model.train()
    total_loss, steps = 0, 0
    t0   = time.time()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

    for ids, mask, lbls in pbar:
        ids, mask, lbls = ids.to(device), mask.to(device), lbls.to(device)
        optimizer.zero_grad()
        loss = focal_loss_fn(model(input_ids=ids, attention_mask=mask).logits, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item(); steps += 1
        pbar.set_postfix({"loss": f"{total_loss/steps:.4f}"})

    avg_loss = total_loss / steps
    y_vt, y_vp = evaluate_loader(model, val_loader)
    val_acc = accuracy_score(y_vt, y_vp)
    val_f1  = f1_score(y_vt, y_vp, average="weighted")

    history["train_loss"].append(avg_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    print(f"  Epoch {epoch+1} | loss={avg_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | {time.time()-t0:.0f}s")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch + 1
        MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model_focal.pt"))
        print(f"  [v] Best model saved (epoch {best_epoch})")

print(f"\nBest: epoch {best_epoch}  val_acc={best_val_acc:.4f}")

model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model_focal.pt"), map_location=device, weights_only=True))
print("\n" + "="*70)
print("TEST RESULTS -- TruthSeeker (in-domain)")
print("="*70)
y_true, y_pred = evaluate_loader(model, test_loader)
r_mb = get_metrics(y_true, y_pred, "DistilBERT + Focal Loss (PROPOSED)")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Fake(0)", "Real(1)"]))

r_mb_cd = None
if ISOT_AVAILABLE:
    print("\n" + "="*70)
    print("CROSS-DOMAIN TEST -- ISOT")
    print("="*70)
    isot_ds     = tokenize_all(X_isot, y_isot, tokenizer, MAX_LEN)
    isot_loader = DataLoader(isot_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    y_isot_true, y_isot_pred = evaluate_loader(model, isot_loader)
    r_mb_cd = get_metrics(y_isot_true, y_isot_pred, "DistilBERT + Focal Loss -> ISOT (cross-domain)")
    print("\nClassification Report (ISOT):")
    print(classification_report(y_isot_true, y_isot_pred, target_names=["Fake(0)", "Real(1)"]))


PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ep = range(1, EPOCHS+1)
ax1.plot(ep, history["train_loss"], "o-", color="#e74c3c", linewidth=2)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Focal Loss"); ax1.set_title("Training Loss"); ax1.grid(alpha=0.3)
ax2.plot(ep, history["val_acc"], "o-", color="#3498db", linewidth=2, label="Val Accuracy")
ax2.plot(ep, history["val_f1"],  "s--", color="#2ecc71", linewidth=2, label="Val F1")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score"); ax2.set_title("Validation Metrics")
ax2.legend(); ax2.grid(alpha=0.3)
plt.suptitle("DistilBERT + Focal Loss -- Training Curves", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "learning_curve.png"), dpi=130); plt.close()

compare_labels = ["SVM+TF-IDF\n(Base Paper)", "MLP+TF-IDF\n(Base Paper)", "DistilBERT\n+Focal Loss\n(Ours)"]
compare_accs   = [BASE_SVM_ACC, BASE_MLP_ACC, r_mb["accuracy"]]
compare_f1s    = [BASE_SVM_ACC, BASE_MLP_ACC, r_mb["f1"]]
colors = ["#95a5a6", "#7f8c8d", "#e74c3c"]
x = np.arange(3); w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x-w/2, compare_accs, w, label="Accuracy", color=colors, alpha=0.9, edgecolor="black", linewidth=0.7)
ax.bar(x+w/2, compare_f1s,  w, label="F1 Score",  color=colors, alpha=0.6, edgecolor="black", linewidth=0.7)
ax.set_xticks(x); ax.set_xticklabels(compare_labels, fontsize=9)
ax.set_ylim(0.92, 1.02); ax.set_ylabel("Score")
ax.set_title("DistilBERT + Focal Loss vs Base Paper Methods")
ax.legend(); ax.grid(axis="y", alpha=0.3)
for i,(a,f) in enumerate(zip(compare_accs,compare_f1s)):
    ax.text(i-w/2, a+0.001, f"{a:.4f}", ha="center", fontsize=8, fontweight="bold")
    ax.text(i+w/2, f+0.001, f"{f:.4f}", ha="center", fontsize=8, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=130); plt.close()

if r_mb_cd:
    fig, ax = plt.subplots(figsize=(6,5))
    bars = ax.bar(["In-Domain\n(TruthSeeker)", "Cross-Domain\n(ISOT)"],
                  [r_mb["accuracy"], r_mb_cd["accuracy"]],
                  color=["#e74c3c","#c0392b"], edgecolor="black", linewidth=0.7, width=0.4)
    ax.set_ylim(max(0, min(r_mb["accuracy"],r_mb_cd["accuracy"])-0.1), 1.02)
    ax.set_ylabel("Accuracy"); ax.set_title("Cross-Domain Generalization"); ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, [r_mb["accuracy"], r_mb_cd["accuracy"]]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cross_domain.png"), dpi=130); plt.close()

cm   = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Fake(0)","Real(1)"])
fig, ax = plt.subplots(figsize=(5,4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix (TruthSeeker)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=130); plt.close()
print("[v] All plots saved to plots_assignment3/")

rows = [
    dict(model="SVM+TF-IDF (Base Paper)", source="paper", accuracy=BASE_SVM_ACC, f1=BASE_SVM_ACC),
    dict(model="MLP+TF-IDF (Base Paper)", source="paper", accuracy=BASE_MLP_ACC, f1=BASE_MLP_ACC),
    dict(model="DistilBERT+FocalLoss (in-domain)", source="ours", accuracy=r_mb["accuracy"], f1=r_mb["f1"]),
]
if r_mb_cd:
    rows.append(dict(model="DistilBERT+FocalLoss (cross-domain)", source="ours",
                     accuracy=r_mb_cd["accuracy"], f1=r_mb_cd["f1"]))
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results.csv"), index=False)

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"  Base paper SVM+TF-IDF (reported) : {BASE_SVM_ACC*100:.2f}%")
print(f"  Base paper MLP+TF-IDF (reported) : {BASE_MLP_ACC*100:.2f}%")
print(f"  DistilBERT + Focal Loss (ours)   : {r_mb['accuracy']*100:.2f}%  <- PROPOSED")
if r_mb_cd:
    print(f"  Cross-domain ISOT                : {r_mb_cd['accuracy']*100:.2f}%")
print("\n[v] All done!")