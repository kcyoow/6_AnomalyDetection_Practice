"""FC-75 글자 획만 비교하는 템플릿 매칭으로 기판 종류 라벨을 다시 확인한다.

기존 방식(variant_score.py)은 템플릿 사각형 전체(글자 + 파란 바탕)를 비교해서, 결함이나 스티커가 인쇄 일부를
가리면 점수가 크게 떨어졌다. 여기서는 템플릿에서 흰 글자 획만 마스크로 남겨(Otsu 이진화 후 1px 팽창)
글자 부분만 비교한다. 회전(±10°), 절반 해상도, 가장자리 10% 자르기는 기존 방식과 같다.

usage: textmask_check.py <VisA root> <evidence dir with fc75_match_all.csv>
"""
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCALE = 0.5
ANGLES = range(-10, 11, 2)


def masked_rotations(tpl, mask):
    h, w = tpl.shape
    out = []
    for a in ANGLES:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        rt = cv2.warpAffine(tpl, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        rm = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST)
        my, mx = int(h * 0.1), int(w * 0.1)
        out.append((rt[my:h - my, mx:w - mx], rm[my:h - my, mx:w - mx]))
    return out


def main():
    root, ev = Path(sys.argv[1]), Path(sys.argv[2])
    tpl = cv2.imread(str(Path(__file__).resolve().parent / "fc75.png"), cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(tpl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 흰 글자 획
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    t = cv2.resize(tpl, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_NEAREST)
    rots = masked_rotations(t, m)

    df = pd.read_csv(ev / "fc75_match_all.csv")
    scores = []
    for r in df.itertuples():
        g = cv2.resize(cv2.imread(str(root / r.image), cv2.IMREAD_GRAYSCALE), None, fx=SCALE, fy=SCALE,
                       interpolation=cv2.INTER_AREA)
        best = -1.0
        for rt, rm in rots:
            res = np.nan_to_num(cv2.matchTemplate(g, rt, cv2.TM_CCOEFF_NORMED, mask=rm), nan=-1, posinf=-1, neginf=-1)
            best = max(best, float(res.max()))
        scores.append(round(best, 4))
    df["textmask_score"] = scores
    df["id"] = df.image.map(lambda p: p.split("/")[-2][0] + p.split("/")[-1].split(".")[0])
    df[["image", "label", "fc75", "source", "fc75_score_new", "textmask_score"]].to_csv(
        ev / "fc75_textmask_scores.csv", index=False)

    thr = (df[~df.fc75].textmask_score.max() + df[df.fc75].textmask_score.min()) / 2
    disagree = df[(df.textmask_score >= thr) != df.fc75]
    old_wrong = df[(df.fc75_score_new >= 0.7) != df.fc75]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, col, cut, title in [(axes[0], "fc75_score_new", 0.7, "whole template (current)"),
                                (axes[1], "textmask_score", thr, "text strokes only (mask)")]:
        bins = np.linspace(0.2, 1.0, 81)
        ax.hist(df[~df.fc75][col], bins=bins, alpha=0.7, label="labelled non-FC-75")
        ax.hist(df[df.fc75][col], bins=bins, alpha=0.7, label="labelled FC-75")
        for _, rr in df[df.source == "visual"].iterrows():
            ax.annotate(rr.id, (rr[col], 1), xytext=(rr[col], 6 + 4 * (rr.id in ("A039", "A093"))), fontsize=7,
                        ha="center", arrowprops=dict(arrowstyle="-", lw=0.5))
        ax.axvline(cut, color="k", ls="--", lw=1, label=f"threshold {cut:.2f}")
        ax.set(title=title, xlabel="match score", yscale="log")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("images")
    plt.tight_layout()
    plt.savefig(ev / "fc75_textmask_compare.png", dpi=130)
    print({"threshold": round(thr, 3), "disagree_with_labels": disagree.id.tolist(),
           "current_method_wrong": old_wrong.id.tolist(),
           "visual_rows": df[df.source == "visual"][["id", "fc75", "fc75_score_new", "textmask_score"]].values.tolist()})


if __name__ == "__main__":
    main()
