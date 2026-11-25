"""자료 분리 점검: 같은 물체 그룹이 두 세트에 걸쳐 있지 않은지, 세트별 사례 수를 확인한다.

근거 파일: data_split_review/dataset_splits.csv (사진별 group_id·split·제외 사유)
예) python check_split.py --data ../TrashNet_dataset
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from common import CLASSES, collect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../TrashNet_dataset")
    ap.add_argument("--manifest", default="data_split_review/dataset_splits.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8-sig")))
    kept = [r for r in rows if r["status"] == "keep"]
    excluded = [r for r in rows if r["status"] != "keep"]
    print(f"원본 사진 {len(rows)}장 → 사용 {len(kept)}장, 제외 {len(excluded)}장")
    for r in excluded:
        print(f"  제외: {r['path']} ({r['reason']})")

    group_splits = defaultdict(set)
    for r in kept:
        group_splits[r["group_id"]].add(r["split"])
    crossing = [g for g, s in group_splits.items() if len(s) > 1]
    sizes = Counter(r["group_id"] for r in kept)
    print(f"물체 그룹 수: {len(group_splits)}  (2장 이상인 그룹 {sum(v > 1 for v in sizes.values())}개, "
          f"가장 큰 그룹 {max(sizes.values())}장)")
    print(f"두 세트 이상에 걸친 그룹: {len(crossing)}개" + ("  ✔ 누수 없음" if not crossing else f" ✘ {crossing[:10]}"))

    print("\n세트별 사례 수 (실제 폴더 기준)")
    print(f"{'split':6s}" + "".join(f"{c:>10s}" for c in CLASSES) + f"{'합계':>8s}")
    for s in ["train", "val", "test"]:
        labels = [l for _, l in collect(Path(args.data) / s)]
        cnt = Counter(labels)
        n_groups = len({r["group_id"] for r in kept if r["split"] == s})
        print(f"{s:6s}" + "".join(f"{cnt[i]:>10d}" for i in range(len(CLASSES)))
              + f"{len(labels):>9d}   (그룹 {n_groups}개)")


if __name__ == "__main__":
    main()
