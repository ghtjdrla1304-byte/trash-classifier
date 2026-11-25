"""학습된 모델로 사진을 예측하거나, 새로 찍은 사진 폴더로 추가 시험을 한다.

예)
  # 사진 한 장 또는 폴더 예측
  python predict.py --model checkpoints/efficientnet_b0.pt --input photo.jpg

  # 추가 시험: 학습에 없던 물체를 다른 배경에서 찍어 <폴더>/<클래스>/ 에 넣고 평가
  python predict.py --model checkpoints/efficientnet_b0.pt --evaluate my_photos/
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from common import IMG_EXT, KOREAN, collect, get_transforms, load_checkpoint


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/efficientnet_b0.pt")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="이미지 파일 또는 이미지가 든 폴더")
    g.add_argument("--evaluate", help="<폴더>/<클래스 이름>/*.jpg 구조의 추가 시험 폴더")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.model, device)
    classes = ckpt["classes"]
    _, tf = get_transforms(ckpt.get("size", 224))

    def probs_of(path):
        x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        return torch.softmax(model(x), 1)[0].cpu().numpy()

    if args.input:
        src = Path(args.input)
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
        for f in files:
            pr = probs_of(f)
            k = int(pr.argmax())
            detail = "  ".join(f"{KOREAN[c]} {pr[i]:.1%}" for i, c in enumerate(classes))
            print(f"{f.name}: {KOREAN[classes[k]]} ({classes[k]}) {pr[k]:.1%}  |  {detail}")
        return

    items = collect(args.evaluate, classes, allow_missing=True)
    if not items:
        raise SystemExit("평가할 사진이 없습니다. <폴더>/<클래스 이름>/ 구조인지 확인하세요.")
    y = np.array([l for _, l in items])
    p = np.array([int(probs_of(path).argmax()) for path, _ in items])
    print(f"추가 시험 정확도: {(y == p).mean():.4f} (n={len(y)})")
    present = sorted(set(y.tolist()) | set(p.tolist()))
    print(classification_report(y, p, labels=present, target_names=[classes[i] for i in present],
                                digits=3, zero_division=0))
    print("혼동행렬 (행=실제, 열=예측):", [classes[i] for i in present])
    print(confusion_matrix(y, p, labels=present))
    for (path, _), t, q in zip(items, y, p):
        if t != q:
            print(f"  틀림 {Path(path).name}: 실제 {KOREAN[classes[t]]} → 예측 {KOREAN[classes[q]]}")


if __name__ == "__main__":
    main()
