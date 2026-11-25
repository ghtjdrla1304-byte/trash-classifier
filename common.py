"""공통 모듈: 클래스 정의, 데이터셋, 변환, 모델 구성."""
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
KOREAN = {"cardboard": "골판지", "glass": "유리", "metal": "금속",
          "paper": "종이", "plastic": "플라스틱", "trash": "일반쓰레기"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def collect(folder, classes=CLASSES, allow_missing=False):
    """<folder>/<클래스>/*.jpg 구조에서 (경로, 라벨) 목록을 만든다. 정답 = 폴더 이름."""
    items = []
    for label, name in enumerate(classes):
        d = Path(folder) / name
        if not d.is_dir():
            if allow_missing:
                continue
            raise FileNotFoundError(f"클래스 폴더가 없습니다: {d}")
        files = sorted(p for p in d.iterdir()
                       if p.suffix.lower() in IMG_EXT and not p.name.startswith("."))
        items += [(str(p), label) for p in files]
    return items


def get_transforms(size=224):
    # 증강(뒤집기·회전·밝기 변화)은 학습 데이터에만 적용한다.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


class TrashDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class SimpleCNN(nn.Module):
    """비교용: 사전학습 없이 처음부터 학습하는 작은 CNN (합성곱 4층)."""

    def __init__(self, num_classes):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


ARCHS = ["efficientnet_b0", "convnext_tiny", "resnet18", "mobilenet_v3", "simple_cnn"]


def build_model(arch, num_classes, pretrained=True):
    if arch == "efficientnet_b0":
        # 깊이·너비·해상도를 함께 키운 효율적인 CNN (6분류 기준 약 400만 파라미터)
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(0.3),
                                     nn.Linear(m.classifier[-1].in_features, num_classes))
    elif arch == "convnext_tiny":
        # Transformer 설계를 CNN에 옮긴 최신 구조 (약 2,800만 파라미터)
        w = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = models.convnext_tiny(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenet_v3":
        w = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_large(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "simple_cnn":
        m = SimpleCNN(num_classes)
    else:
        raise ValueError(arch)
    return m


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
