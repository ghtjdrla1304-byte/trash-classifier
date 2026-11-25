"""쓰레기 분류 — ConvNeXt-Tiny (단독 실행 파일)

이 파일 하나만 있으면 학습·평가·예측을 모두 할 수 있습니다. 다른 파일이 필요 없습니다.

[분류 클래스 6개]
  cardboard(골판지) / glass(유리) / metal(금속) / paper(종이) / plastic(플라스틱) / trash(일반쓰레기)

[모델]
  ConvNeXt-Tiny: Vision Transformer의 설계(큰 7x7 커널, LayerNorm, GELU, 적은 정규화층)를
  CNN에 옮겨 온 최신 구조. ImageNet으로 미리 학습된 가중치에서 시작해 마지막 분류층만
  6개 클래스로 바꾸고 전체를 미세조정합니다.
  파라미터 약 2,800만 개(EfficientNet-B0의 약 7배), GPU 메모리를 더 쓰므로 batch 16을 기본으로 함.
  우리 팀 결과: 테스트 정확도 92.6%, macro-F1 0.910, 검증 macro-F1 0.934 (TrashNet 그룹 분할, 377장)

[데이터 폴더 구조] — 같은 물체 사진은 한 세트에만 들어가도록 미리 나눠 둔 폴더
  TrashNet_dataset/
    train/cardboard/*.jpg, train/glass/*.jpg, ...
    val/...
    test/...

[필요 패키지]
  pip install torch torchvision scikit-learn matplotlib pillow numpy

[사용법]
  # 1) 학습 + 기준선 비교 + 테스트 평가
  python trash_convnext_tiny.py train --data ./TrashNet_dataset

  # 2) 사진 예측 (파일 하나 또는 폴더)
  python trash_convnext_tiny.py predict --input photo.jpg

  # 3) 새로 찍은 사진으로 추가 시험 (폴더 안에 클래스 이름의 하위 폴더)
  python trash_convnext_tiny.py evaluate --input ./my_photos

  Colab: 런타임 → 런타임 유형 변경 → GPU 선택 후
         !python trash_convnext_tiny.py train --data /content/TrashNet_dataset
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ------------------------------------------------------------------ 설정 ---
MODEL_NAME = "convnext_tiny"
DEFAULTS = dict(epochs=20, lr=1e-4, batch=16)  # 우리 팀이 사용한 값

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
KOREAN = {"cardboard": "골판지", "glass": "유리", "metal": "금속",
          "paper": "종이", "plastic": "플라스틱", "trash": "일반쓰레기"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet 정규화 값
IMG_SIZE = 224


# ------------------------------------------------------------------ 모델 ---
def build_model(num_classes=len(CLASSES), pretrained=True):
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    # classifier = [LayerNorm2d, Flatten, Linear(768 → 1000)]
    in_features = model.classifier[-1].in_features  # 768
    # 원래 1000개 클래스(ImageNet)용 마지막 Linear 층만 6개 클래스용으로 교체
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------- 데이터 ---
def get_transforms():
    # 증강(자르기·뒤집기·회전·밝기 변화)은 학습 데이터에만 적용
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def collect(folder, allow_missing=False):
    """<folder>/<클래스 이름>/*.jpg → [(경로, 라벨번호)]. 정답은 폴더 이름."""
    items = []
    for label, name in enumerate(CLASSES):
        d = Path(folder) / name
        if not d.is_dir():
            if allow_missing:
                continue
            raise FileNotFoundError(f"클래스 폴더가 없습니다: {d}")
        items += [(str(p), label) for p in sorted(d.iterdir())
                  if p.suffix.lower() in IMG_EXT and not p.name.startswith(".")]
    return items


class TrashDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label


# ---------------------------------------------------------------- 공통 ---
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu())
        labels.append(y)
    probs = torch.cat(probs).numpy()
    return torch.cat(labels).numpy(), probs.argmax(1), probs


def print_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    names = [KOREAN[c] for c in CLASSES]
    print("\n혼동행렬 (행=실제 정답, 열=모델 예측)")
    print(" " * 10 + "".join(f"{n:>7}" for n in names) + "   재현율")
    for i, n in enumerate(names):
        print(f"{n:<8}" + "".join(f"{v:>8}" for v in cm[i]) + f"   {cm[i, i] / max(cm[i].sum(), 1):.1%}")
    return cm


def load_trained(path, device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


# ---------------------------------------------------------------- 학습 ---
def train(args):
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = {s: collect(Path(args.data) / s) for s in ["train", "val", "test"]}
    for s, items in splits.items():
        cnt = np.bincount([l for _, l in items], minlength=len(CLASSES))
        print(f"{s:5s} {len(items):4d}장 | " + " ".join(f"{c}={n}" for c, n in zip(CLASSES, cnt)))
    print(f"모델={MODEL_NAME}  device={device}")

    train_tf, eval_tf = get_transforms()
    kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=device == "cuda")
    train_dl = DataLoader(TrashDataset(splits["train"], train_tf), shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(TrashDataset(splits["val"], eval_tf), **kw)
    test_dl = DataLoader(TrashDataset(splits["test"], eval_tf), **kw)

    # 클래스 불균형 보정: 적은 클래스(일반쓰레기)의 오답에 더 큰 벌점. 학습셋 개수로만 계산.
    tr_labels = np.array([l for _, l in splits["train"]])
    counts = np.bincount(tr_labels, minlength=len(CLASSES))
    class_w = torch.tensor(len(tr_labels) / (len(CLASSES) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.1)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_dl), pct_start=0.15)

    best_f1, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        n = correct = 0
        loss_sum = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)

        yv, pv, _ = predict_loader(model, val_dl, device)
        val_acc, val_f1 = float((yv == pv).mean()), float(f1_score(yv, pv, average="macro"))
        history.append(dict(epoch=epoch, train_loss=loss_sum / n, train_acc=correct / n,
                            val_acc=val_acc, val_f1=val_f1))
        mark = ""
        if val_f1 > best_f1:  # 가장 좋은 모델은 검증셋(val)으로만 고름 — test는 마지막에 한 번만
            best_f1 = val_f1
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "model": MODEL_NAME,
                        "epoch": epoch}, args.out)
            mark = "  ← 저장"
        print(f"epoch {epoch:02d}/{args.epochs}  loss={loss_sum / n:.4f}  train_acc={correct / n:.3f}"
              f"  val_acc={val_acc:.3f}  val_F1={val_f1:.3f}{mark}")

    # ---- 최종 시험: 테스트셋은 여기서 단 한 번 사용 ----
    model = load_trained(args.out, device)
    yt, pt, _ = predict_loader(model, test_dl, device)
    majority = int(counts.argmax())
    base = np.full_like(yt, majority)
    print("\n===== 같은 테스트셋에서 비교 =====")
    print(f"기준선 (항상 '{KOREAN[CLASSES[majority]]}'): 정확도 {(yt == base).mean():.4f}  "
          f"macro-F1 {f1_score(yt, base, average='macro'):.4f}")
    print(f"{MODEL_NAME:28s}: 정확도 {(yt == pt).mean():.4f}  macro-F1 {f1_score(yt, pt, average='macro'):.4f}")
    print(classification_report(yt, pt, target_names=CLASSES, digits=3, zero_division=0))
    cm = print_confusion(yt, pt)

    result = dict(model=MODEL_NAME, test_acc=float((yt == pt).mean()),
                  test_macro_f1=float(f1_score(yt, pt, average="macro")),
                  baseline_acc=float((yt == base).mean()), best_val_f1=best_f1,
                  confusion_matrix=cm.tolist(), history=history)
    out_json = Path(args.out).with_suffix(".json")
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n모델 저장: {args.out}   결과 저장: {out_json}")


# ------------------------------------------------------------ 평가/예측 ---
def evaluate(args):
    """학습에 쓰지 않은 새 사진 폴더(<폴더>/<클래스 이름>/)로 추가 시험."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained(args.model, device)
    items = collect(args.input, allow_missing=True)
    if not items:
        raise SystemExit("사진이 없습니다. <폴더>/<클래스 이름>/*.jpg 구조인지 확인하세요.")
    _, eval_tf = get_transforms()
    y, p, _ = predict_loader(model, DataLoader(TrashDataset(items, eval_tf), batch_size=32), device)
    print(f"추가 시험 정확도: {(y == p).mean():.4f} (n={len(y)})")
    print_confusion(y, p)
    for (path, _), t, q in zip(items, y, p):
        if t != q:
            print(f"  틀림 {Path(path).name}: 실제 {KOREAN[CLASSES[t]]} → 예측 {KOREAN[CLASSES[q]]}")


@torch.no_grad()
def predict(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained(args.model, device)
    _, eval_tf = get_transforms()
    src = Path(args.input)
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    for f in files:
        prob = torch.softmax(model(eval_tf(Image.open(f).convert("RGB")).unsqueeze(0).to(device)), 1)[0].cpu()
        k = int(prob.argmax())
        detail = "  ".join(f"{KOREAN[c]} {prob[i]:.1%}" for i, c in enumerate(CLASSES))
        print(f"{f.name}: {KOREAN[CLASSES[k]]} ({prob[k]:.1%})  |  {detail}")


def main():
    ap = argparse.ArgumentParser(description=f"쓰레기 분류 — {MODEL_NAME}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    default_ckpt = f"trash_{MODEL_NAME}.pt"

    t = sub.add_parser("train", help="학습 + 테스트 평가")
    t.add_argument("--data", default="./TrashNet_dataset", help="train/val/test 폴더가 있는 경로")
    t.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    t.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    t.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--out", default=default_ckpt)
    t.set_defaults(fn=train)

    e = sub.add_parser("evaluate", help="새 사진 폴더로 추가 시험")
    e.add_argument("--model", default=default_ckpt)
    e.add_argument("--input", required=True)
    e.set_defaults(fn=evaluate)

    p = sub.add_parser("predict", help="사진 예측")
    p.add_argument("--model", default=default_ckpt)
    p.add_argument("--input", required=True, help="이미지 파일 또는 폴더")
    p.set_defaults(fn=predict)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
