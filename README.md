# 쓰레기 분류 (TrashNet 6분류) — 인공지능과 머신러닝 텀프로젝트 ④

사진 한 장을 보고 물체의 재질을 **골판지 / 유리 / 금속 / 종이 / 플라스틱 / 일반쓰레기** 중 하나로 분류합니다.

## 1. 문제와 데이터

| 항목 | 내용 |
|---|---|
| 출처 | TrashNet — https://github.com/garythung/trashnet (`dataset-resized`, 512×384) |
| 사례 단위 | 사진 1장 |
| 입력 | RGB 사진 (224×224로 크기 조정) |
| 정답 | 사진이 들어 있는 폴더 이름 (예: `paper/paper1.jpg` → 종이) |
| 원본 수 | 2,527장 (cardboard 403, glass 501, metal 410, paper 594, plastic 482, trash 137) |
| 결측·중복 처리 | 라벨 오류로 확인된 5장 제외 (`data_split_review/excluded_files.csv`), 같은 물체를 여러 각도로 찍은 사진은 그룹으로 묶음 |

## 2. 자료 분리와 누수 방지

TrashNet에는 **같은 물체를 각도·조명만 바꿔 여러 번 찍은 사진**이 많습니다. 사진 단위로 무작위 분할하면 거의 같은 사진이 학습과 시험에 함께 들어가 정확도가 부풀려집니다.

- 이미지 유사도 후보 + 사람 검토(`pair_review.csv`, `group_edges.csv`, `manual_groups.csv`)로 **같은 물체 그룹**을 만들었습니다.
- 그룹 단위로 train / val / test(약 70/15/15)에 나눴고, **한 그룹은 한 세트에만** 들어갑니다 (`dataset_splits.csv`, `object_groups.csv`).
- 증강(뒤집기·회전·밝기 변경)은 분할 **이후 학습 데이터에만** 실시간 적용합니다.
- 모델 선택(최고 epoch)은 **val**의 macro-F1로만 하고, **test는 마지막에 한 번만** 평가합니다.

| split | 골판지 | 유리 | 금속 | 종이 | 플라스틱 | 일반쓰레기 | 합계 |
|---|---|---|---|---|---|---|---|
| train | 282 | 350 | 286 | 414 | 337 | 96 | 1,765 |
| val | 61 | 75 | 62 | 89 | 72 | 21 | 380 |
| test | 60 | 75 | 61 | 89 | 72 | 20 | 377 |

`python check_split.py` 로 그룹이 세트 사이에 겹치지 않음을 직접 확인할 수 있습니다.

## 3. 결과

RESULTS_PLACEHOLDER

## 4. 재현 방법 (Readme)

```bash
# 1) 코드 받기
git clone <이 저장소 주소>
cd trash-classifier

# 2) 패키지 설치 (GPU가 있으면 https://pytorch.org 에서 CUDA 버전 torch 설치 권장)
pip install -r requirements.txt

# 3) 데이터 준비: 그룹 분할된 TrashNet_dataset 폴더를 저장소 옆(../TrashNet_dataset)에 둔다
#    ../TrashNet_dataset/{train,val,test}/{cardboard,glass,metal,paper,plastic,trash}/*.jpg
#    (다른 위치라면 --data 경로로 지정)

# 4) 분할 점검
python check_split.py --data ../TrashNet_dataset

# 5) 학습 + 기준선 비교 + 테스트 평가 (결과는 results/<모델>/ 에 저장)
python train.py --arch resnet18     --epochs 20 --lr 3e-4
python train.py --arch mobilenet_v3 --epochs 20 --lr 5e-4
python train.py --arch simple_cnn   --epochs 40 --lr 3e-3

# 6) 예측 / 추가 시험(학습에 없던 물체를 다른 배경에서 찍은 사진)
python predict.py --model checkpoints/resnet18.pt --input photo.jpg
python predict.py --model checkpoints/resnet18.pt --evaluate my_photos/   # my_photos/<클래스>/*.jpg
```

Colab에서 실행할 때는 저장소와 `TrashNet_dataset` 폴더를 올린 뒤 런타임을 GPU로 바꾸고 같은 명령을 `!python ...` 으로 실행하면 됩니다.

## 5. 파일 구성

| 파일 | 역할 |
|---|---|
| `common.py` | 클래스 목록, 데이터셋, 증강, 모델(ResNet18 / MobileNetV3 / 작은 CNN) |
| `train.py` | 학습, 기준선(가장 흔한 클래스) 비교, 테스트 평가, 혼동행렬·오답 이미지 저장 |
| `predict.py` | 사진 예측, 새 사진 폴더로 추가 시험 |
| `check_split.py` | 그룹 누수 점검과 세트별 사례 수 |
| `data_split_review/` | 그룹 분할 근거 CSV (사진별 그룹·세트, 제외 사진, 검토 기록) |
| `results/<모델>/` | `metrics.json`, `confusion_matrix.png/csv`, `training_curve.png`, `test_errors.png/csv` |
