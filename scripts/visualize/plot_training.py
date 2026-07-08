# scripts/visualize/plot_training.py
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

sns.set(style="whitegrid", font_scale=1.05)
MODEL_PATTERNS = {
    "ResNet18": re.compile(r"resnet", re.I),
    "EfficientNetB4": re.compile(r"efficientnet|eff", re.I),
    "MobileNetV3": re.compile(r"mobilenet|mobile", re.I),
}

COLORS = {"ResNet18":"#1f77b4","EfficientNetB4":"#ff7f0e","MobileNetV3":"#2ca02c","other":"#8c564b"}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def detect_model_name(j, filename):
    if isinstance(j, dict) and "model_name" in j:
        return j["model_name"]
    for k in ("config","cfg","params"):
        if isinstance(j, dict) and k in j and isinstance(j[k],dict) and "model_name" in j[k]:
            return j[k]["model_name"]
    name = Path(filename).stem
    for display, pattern in MODEL_PATTERNS.items():
        if pattern.search(name):
            return display
    return "other"

def extract_history(j):
    if isinstance(j, dict) and "history" in j:
        return j["history"]
    if isinstance(j, list):
        return j
    for v in j.values() if isinstance(j, dict) else []:
        if isinstance(v, list):
            return v
    return []

def get_series(hist, keys):
    for k in keys:
        if all(isinstance(e, dict) and k in e for e in hist):
            return np.array([e[k] for e in hist], dtype=float)
    return None

def pad_stack(arrs):
    if not arrs: return None
    maxlen = max(a.shape[0] for a in arrs)
    stacked = np.full((len(arrs), maxlen), np.nan)
    for i,a in enumerate(arrs): stacked[i,:a.shape[0]] = a
    return stacked

def plot_by_model(groups, out_dir):
    # prepare figure with two subplots (loss above, accuracy below)
    fig, axes = plt.subplots(2,1, figsize=(10,10), sharex=True)
    ax_loss, ax_acc = axes

    max_epoch = 0
    # prepare data containers for axis autoscaling
    for name, files in groups.items():
        runs_train_loss, runs_val_loss = [], []
        runs_train_acc, runs_val_acc = [], []
        for p in files:
            j = load_json(p)
            hist = extract_history(j)
            train_loss = get_series(hist, ["train_loss","loss","train/loss"])
            val_loss = get_series(hist, ["val_loss","validation_loss","val/loss"])
            train_acc = get_series(hist, ["train_acc","train_accuracy","accuracy","acc"])
            val_acc = get_series(hist, ["val_accuracy","val_acc","validation_accuracy","val/accuracy"])
            if val_acc is None:
                val_acc = get_series(hist, ["val_macro_f1","val_f1","macro_f1"])
            if train_loss is not None: runs_train_loss.append(train_loss); max_epoch = max(max_epoch, train_loss.shape[0])
            if val_loss is not None: runs_val_loss.append(val_loss); max_epoch = max(max_epoch, val_loss.shape[0])
            if train_acc is not None: runs_train_acc.append(train_acc); max_epoch = max(max_epoch, train_acc.shape[0])
            if val_acc is not None: runs_val_acc.append(val_acc); max_epoch = max(max_epoch, val_acc.shape[0])

        # plot individual runs faded
        color = COLORS.get(name,"#333333")
        for r in runs_train_loss: ax_loss.plot(np.arange(1,r.shape[0]+1), r, color=color, alpha=0.18)
        for r in runs_val_loss: ax_loss.plot(np.arange(1,r.shape[0]+1), r, color=color, alpha=0.12, linestyle="--")

        for r in runs_train_acc: ax_acc.plot(np.arange(1,r.shape[0]+1), r, color=color, alpha=0.18)
        for r in runs_val_acc: ax_acc.plot(np.arange(1,r.shape[0]+1), r, color=color, alpha=0.12, linestyle="--")

        # plot mean ± std
        st = pad_stack(runs_train_loss); sv = pad_stack(runs_val_loss)
        sat = pad_stack(runs_train_acc); sav = pad_stack(runs_val_acc)
        if st is not None:
            mean = np.nanmean(st,axis=0); std = np.nanstd(st,axis=0)
            epochs = np.arange(1, mean.shape[0]+1)
            ax_loss.plot(epochs, mean, color=color, linewidth=2.4, label=f"{name} (n={st.shape[0]}) train")
            ax_loss.fill_between(epochs, mean-std, mean+std, color=color, alpha=0.12)
        if sv is not None:
            mean = np.nanmean(sv,axis=0); std = np.nanstd(sv,axis=0)
            epochs = np.arange(1, mean.shape[0]+1)
            ax_loss.plot(epochs, mean, color=color, linewidth=2.4, linestyle="--", label=f"{name} (n={sv.shape[0]}) val")
            ax_loss.fill_between(epochs, mean-std, mean+std, color=color, alpha=0.08)

        if sat is not None:
            mean = np.nanmean(sat,axis=0); std = np.nanstd(sat,axis=0)
            epochs = np.arange(1, mean.shape[0]+1)
            ax_acc.plot(epochs, mean, color=color, linewidth=2.4, label=f"{name} (n={sat.shape[0]}) train")
            ax_acc.fill_between(epochs, mean-std, mean+std, color=color, alpha=0.12)
        if sav is not None:
            mean = np.nanmean(sav,axis=0); std = np.nanstd(sav,axis=0)
            epochs = np.arange(1, mean.shape[0]+1)
            ax_acc.plot(epochs, mean, color=color, linewidth=2.4, linestyle="--", label=f"{name} (n={sav.shape[0]}) val")
            ax_acc.fill_between(epochs, mean-std, mean+std, color=color, alpha=0.08)

    # formatting
    ax_loss.set_title("Train / Val Loss")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_xlim(1, max_epoch)
    ax_loss.grid(True, alpha=0.3)

    ax_acc.set_title("Train / Val Accuracy (or val macro F1 if accuracy missing)")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy / Macro F1")
    ax_acc.set_xlim(1, max_epoch)
    ax_acc.set_ylim(0.45, 1.01)
    ax_acc.grid(True, alpha=0.3)

    # combined legend, sorted by label
    handles, labels = [], []
    for ax in [ax_loss, ax_acc]:
        h,l = ax.get_legend_handles_labels()
        handles += h; labels += l
    # dedupe preserving order
    seen = set(); dedup_h, dedup_l = [], []
    for h,l in zip(handles, labels):
        if l not in seen:
            dedup_h.append(h); dedup_l.append(l); seen.add(l)
    fig.legend(dedup_h, dedup_l, loc="upper center", ncol=2, bbox_to_anchor=(0.5,0.98))
    fig.tight_layout(rect=[0,0,1,0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "train_val_by_model_improved.png", dpi=300)
    print("Saved:", out_dir / "train_val_by_model_improved.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="history json files (default: reports/train_history*.json)")
    parser.add_argument("--out", default="reports", help="output folder")
    args = parser.parse_args()

    if args.files:
        files = [Path(p) for p in args.files]
    else:
        files = sorted(Path("reports").glob("train_history*.json"))

        groups = {}
    for p in files:
        try:
            j = load_json(p)
        except Exception:
            continue
        model = detect_model_name(j, p.name)
        # normalize: nếu JSON có model_name và khớp pattern thì dùng nó,
        # nếu không thì thử dò từ tên file; cuối cùng fallback 'other'
        mkey = "other"
        if isinstance(model, str):
            for key, pat in MODEL_PATTERNS.items():
                if pat.search(str(model)):
                    mkey = key
                    break
        if mkey == "other":
            for key, pat in MODEL_PATTERNS.items():
                if pat.search(p.name):
                    mkey = key
                    break
        groups.setdefault(mkey, []).append(p)

    # ensure groups exist for display order
    for k in ["ResNet18","EfficientNetB4","MobileNetV3","other"]:
        groups.setdefault(k, [])

    # print grouping summary
    print("Grouping summary:")
    for k,v in groups.items():
        print(f" - {k}: {len(v)} files")

    plot_by_model(groups, Path(args.out))

if __name__ == "__main__":
    main()