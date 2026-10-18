"""pcb4 전체 이미지에 대해 인쇄 템플릿(FC-75, HW-168) 매칭 점수를 계산.

usage: variant_score.py <VisA root> <templates dir> <out csv>
templates dir 에는 fc75.png, hw168.png (원본 해상도 기준 그레이 크롭) 가 있어야 함.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SCALE = 0.5  # 속도를 위해 절반 해상도에서 매칭
ANGLES = range(-10, 11, 2)  # 기판이 약간씩 회전되어 있어 템플릿을 돌려가며 매칭


def rotations(tpl):
    h, w = tpl.shape
    out = []
    for a in ANGLES:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(tpl, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        # 회전으로 생긴 가장자리 영향 줄이려고 가운데 80%만 사용
        my, mx = int(h * 0.1), int(w * 0.1)
        out.append(r[my:h - my, mx:w - mx])
    return out


def best_score(gray, tpls):
    best, loc = -1.0, None
    for t in tpls:
        res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        if mx > best:
            best, loc = mx, ml
    return best, loc


def main():
    root, tdir, out_csv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    templates = {}
    for name in [p.stem for p in sorted(tdir.glob("*.png"))]:
        t = cv2.imread(str(tdir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        t = cv2.resize(t, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        templates[name] = rotations(t)

    df = pd.read_csv(root / "pcb4" / "image_anno.csv")
    normal_idx = df.index[df["label"] == "normal"]
    n_train = int(len(normal_idx) * 0.8)  # 노트북과 동일한 분할 (CSV 순서, 앞 80%)
    group = pd.Series("anomaly", index=df.index)
    group[normal_idx[:n_train]] = "train_normal"
    group[normal_idx[n_train:]] = "test_normal"
    df["group"] = group

    rows = []
    for i, r in df.iterrows():
        img = cv2.imread(str(root / r["image"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("missing", r["image"], flush=True)
            continue
        g =cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        rec = {"image": r["image"], "label": r["label"], "group": r["group"]}
        for name, tpls in templates.items():
            s, loc = best_score(g, tpls)
            rec[f"{name}_score"] = round(float(s), 4)
            rec[f"{name}_x"], rec[f"{name}_y"] = int(loc[0] / SCALE), int(loc[1] / SCALE)
        rows.append(rec)
        if i % 100 == 0:
            print(i, rec, flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("saved", out_csv)


if __name__ == "__main__":
    main()
