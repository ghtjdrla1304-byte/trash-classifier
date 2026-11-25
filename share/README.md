# 공유용 단독 실행 파일

각 파일은 **혼자서 동작**합니다. 다른 `.py` 파일 없이, 이 파일 하나와 데이터 폴더만 있으면 학습·평가·예측을 모두 할 수 있습니다.

| 파일 | 모델 | 파라미터 | 기본 설정 | 우리 팀 결과 (테스트 377장) |
|---|---|---|---|---|
| [`trash_efficientnet_b0.py`](trash_efficientnet_b0.py) | EfficientNet-B0 | 약 400만 | 25 epoch, lr 1e-3, batch 32 | 정확도 93.1%, macro-F1 0.920 |
| [`trash_convnext_tiny.py`](trash_convnext_tiny.py) | ConvNeXt-Tiny | 약 2,800만 | 20 epoch, lr 1e-4, batch 16 | 정확도 92.6%, macro-F1 0.910 |

두 파일은 데이터 처리, 증강, 학습 방식, 평가 방식이 모두 같고 **모델 부분(`build_model`)과 기본 하이퍼파라미터만** 다릅니다. 그래서 두 결과를 공정하게 비교할 수 있습니다.

## 준비

```bash
pip install torch torchvision scikit-learn matplotlib pillow numpy
```

데이터는 같은 물체가 한 세트에만 들어가도록 미리 나눈 `TrashNet_dataset` 폴더를 씁니다.

```
TrashNet_dataset/
  train/{cardboard,glass,metal,paper,plastic,trash}/*.jpg
  val/  {...}
  test/ {...}
```

## 실행

```bash
# 학습 + 기준선 비교 + 테스트 평가  →  trash_<모델>.pt, trash_<모델>.json 생성
python trash_efficientnet_b0.py train --data ./TrashNet_dataset
python trash_convnext_tiny.py   train --data ./TrashNet_dataset

# 사진 예측 (파일 또는 폴더)
python trash_efficientnet_b0.py predict --input photo.jpg

# 새로 찍은 사진으로 추가 시험: my_photos/<클래스 이름>/*.jpg
python trash_efficientnet_b0.py evaluate --input ./my_photos
```

- GPU가 없으면 CPU로도 돌아가지만 많이 느립니다. Colab에서는 **런타임 → 런타임 유형 변경 → GPU**를 선택하고 `!python trash_efficientnet_b0.py train --data /content/TrashNet_dataset` 처럼 실행하세요.
- ConvNeXt-Tiny는 GPU 메모리를 더 씁니다. 메모리 부족 오류가 나면 `--batch 8`로 줄이세요.
- epoch·학습률 등은 `--epochs`, `--lr`, `--batch` 옵션으로 바꿀 수 있습니다.
