"""쓰레기 분류 모델 학습 + 기준선 비교 + 테스트 평가 + 오류 분석 저장.

데이터: TrashNet_dataset/{train,val,test}/<클래스>/*.jpg
  - 같은 물체(여러 각도·조명으로 찍은 사진)는 하나의 그룹으로 묶여 한 세트에만 들어가 있음
  - 라벨 오류로 확인된 5장은 제외됨 (review/excluded_files.csv)

예)
  python train.py --arch resnet18 --epochs 20
  python train.py --arch simple_cnn --epochs 40 --lr 3e-3
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from common import ARCHS, CLASSES, KOREAN, TrashDataset, build_model, collect, get_transforms

from matplotlib import font_manager

_fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [f for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"] if f in _fonts] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_all(model, loader, device, amp=False):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        with torch.autocast(device_type=device, enabled=amp):
            out = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(out.float(), 1).cpu())
        labels.append(y)
    probs = torch.cat(probs).numpy()
    return torch.cat(labels).numpy(), probs.argmax(1), probs


def plot_confusion(cm, title, path):
    names = [KOREAN[c] for c in CLASSES]
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(len(names)), names, rotation=30)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("모델 예측")
    ax.set_ylabel("실제 정답")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_history(hist, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, len(hist["train_loss"]) + 1)
    a1.plot(ep, hist["train_loss"], label="train")
    a1.plot(ep, hist["val_loss"], label="val")
    a1.set_title("손실(loss)")
    a1.set_xlabel("epoch")
    a1.legend()
    a2.plot(ep, hist["train_acc"], label="train")
    a2.plot(ep, hist["val_acc"], label="val")
    a2.set_title("정확도")
    a2.set_xlabel("epoch")
    a2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_errors(items, y, p, probs, path, max_n=24):
    """틀린 사진을 확신도가 높은 순으로 모아 보여준다 (오류 분석용)."""
    wrong = [i for i in range(len(y)) if y[i] != p[i]]
    wrong.sort(key=lambda i: -probs[i, p[i]])
    wrong = wrong[:max_n]
    if not wrong:
        return
    cols = 6
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.7))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), wrong):
        ax.imshow(Image.open(items[i][0]).convert("RGB"))
        ax.set_title(f"실제 {KOREAN[CLASSES[y[i]]]}\n예측 {KOREAN[CLASSES[p[i]]]} {probs[i, p[i]]:.0%}",
                     fontsize=9, color="crimson")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../TrashNet_dataset", help="train/val/test 폴더가 있는 경로")
    ap.add_argument("--arch", default="efficientnet_b0", choices=ARCHS)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 혼합정밀도 사용. GTX 16xx 계열에서는 NaN 출력이 생겨 기본값은 끔")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.arch + ("_scratch" if args.no_pretrained and args.arch != "simple_cnn" else "")
    out = Path(args.out_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    ckpt_path = Path("checkpoints") / f"{run_name}.pt"

    splits = {s: collect(Path(args.data) / s) for s in ["train", "val", "test"]}
    for s, it in splits.items():
        cnt = np.bincount([l for _, l in it], minlength=len(CLASSES))
        print(f"{s:5s}: {len(it):4d}장  " + "  ".join(f"{c}={n}" for c, n in zip(CLASSES, cnt)))
    print(f"device={device}  arch={args.arch}  pretrained={not args.no_pretrained}")

    train_tf, eval_tf = get_transforms(args.size)
    kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=device == "cuda",
              persistent_workers=args.workers > 0)
    train_dl = DataLoader(TrashDataset(splits["train"], train_tf), shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(TrashDataset(splits["val"], eval_tf), shuffle=False, **kw)
    test_dl = DataLoader(TrashDataset(splits["test"], eval_tf), shuffle=False, **kw)

    # 클래스 불균형(일반쓰레기 96장) 보정: 학습셋 개수로만 가중치 계산
    tr_labels = np.array([l for _, l in splits["train"]])
    counts = np.bincount(tr_labels, minlength=len(CLASSES))
    weights = torch.tensor(len(tr_labels) / (len(CLASSES) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    model = build_model(args.arch, len(CLASSES), pretrained=not args.no_pretrained).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(train_dl), pct_start=0.15)
    amp = args.amp and device == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]}
    best = -1.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        n = correct = 0
        loss_sum = 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)

        yv, pv, prv = predict_all(model, val_dl, device, amp)
        val_loss = float(-np.log(prv[np.arange(len(yv)), yv] + 1e-8).mean())
        val_acc = float((yv == pv).mean())
        val_f1 = f1_score(yv, pv, average="macro")
        for k, v in zip(hist, [loss_sum / n, correct / n, val_loss, val_acc, val_f1]):
            hist[k].append(v)
        mark = ""
        if val_f1 > best:  # 모델 선택은 검증셋(val)으로만 — 테스트셋은 마지막에 한 번만 사용
            best = val_f1
            torch.save({"state_dict": model.state_dict(), "arch": args.arch, "classes": CLASSES,
                        "size": args.size, "epoch": epoch}, ckpt_path)
            mark = " *"
        print(f"epoch {epoch:02d}/{args.epochs}  train_loss={loss_sum / n:.4f} train_acc={correct / n:.4f}"
              f"  val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}{mark}")
    train_time = time.time() - t0

    # ----- 최종 시험 (테스트셋은 여기서 단 한 번 사용) -----
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    yt, pt, prt = predict_all(model, test_dl, device, amp)
    assert not np.isnan(prt).any(), "모델 출력에 NaN이 있습니다 (--amp 를 끄세요)"
    majority = int(counts.argmax())
    base = np.full_like(yt, majority)
    res = {
        "arch": args.arch, "pretrained": not args.no_pretrained and args.arch != "simple_cnn", "epochs": args.epochs,
        "best_epoch": int(torch.load(ckpt_path, map_location="cpu")["epoch"]),
        "train_time_sec": round(train_time, 1),
        "baseline_majority_class": CLASSES[majority],
        "baseline_test_acc": float((yt == base).mean()),
        "baseline_test_macro_f1": float(f1_score(yt, base, average="macro")),
        "best_val_macro_f1": float(best),
        "test_acc": float((yt == pt).mean()),
        "test_macro_f1": float(f1_score(yt, pt, average="macro")),
        "n_test": int(len(yt)),
    }
    print("\n----- 같은 테스트셋에서 방법 비교 -----")
    print(f"기준선 (항상 '{KOREAN[CLASSES[majority]]}'): acc={res['baseline_test_acc']:.4f} "
          f"macro-F1={res['baseline_test_macro_f1']:.4f}")
    print(f"{args.arch:28s}: acc={res['test_acc']:.4f} macro-F1={res['test_macro_f1']:.4f}")
    rep = classification_report(yt, pt, target_names=CLASSES, digits=3, zero_division=0)
    print(rep)
    cm = confusion_matrix(yt, pt, labels=range(len(CLASSES)))
    print("혼동행렬 (행=실제, 열=예측):", CLASSES)
    print(cm)

    (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "classification_report.txt").write_text(rep, encoding="utf-8")
    (out / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    with open(out / "confusion_matrix.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["실제/예측"] + [KOREAN[c] for c in CLASSES])
        for i, c in enumerate(CLASSES):
            w.writerow([KOREAN[c]] + cm[i].tolist())
    with open(out / "test_errors.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "true", "pred", "confidence"])
        for i in range(len(yt)):
            if yt[i] != pt[i]:
                w.writerow([Path(splits["test"][i][0]).name, CLASSES[yt[i]], CLASSES[pt[i]],
                            f"{prt[i, pt[i]]:.3f}"])
    plot_confusion(cm, f"{args.arch} 테스트 혼동행렬 (acc {res['test_acc']:.1%})", out / "confusion_matrix.png")
    plot_history(hist, out / "training_curve.png")
    plot_errors(splits["test"], yt, pt, prt, out / "test_errors.png")
    print(f"\n결과 저장: {out}/  모델: {ckpt_path}")


if __name__ == "__main__":
    main()
