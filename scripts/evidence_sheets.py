"""기판 종류(FC-75) 라벨의 검증 자료를 만든다.

1,105장 모두에 대해 FC-75 인쇄 템플릿 매칭 점수와 가장 잘 맞은 위치를 다시 구하고,
그 위치를 잘라 격자 이미지로 모은다. 사람이 격자를 보고 라벨을 한 장씩 확인할 수 있다.

usage: evidence_sheets.py <VisA root> <labels csv> <out dir>
"""
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from variant_score import SCALE, rotations  # noqa: E402

THR = 0.7
MARGIN = 26


def match(gray_half, tpls):
    best, loc, size = -1.0, None, None
    for t in tpls:
        res = cv2.matchTemplate(gray_half, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        if mx > best:
            best, loc, size = mx, ml, t.shape
    return best, loc, size


def tile(img, row, w, h, scale):
    x, y = int(row.x / SCALE), int(row.y / SCALE)
    tw, th = int(row.tw / SCALE), int(row.th / SCALE)
    x0, y0 = max(0, x - MARGIN), max(0, y - MARGIN)
    crop = img[y0:y + th + MARGIN, x0:x + tw + MARGIN]
    crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    t = np.full((h, w, 3), 255, np.uint8)
    t[16:16 + min(crop.shape[0], h - 16), :min(crop.shape[1], w)] = crop[:h - 16, :w]
    color = (60, 140, 40) if row.fc75 else (150, 150, 150)
    if row.source == "visual":
        color = (230, 120, 0)  # BGR 파란색: 눈으로 판정한 행
    cv2.rectangle(t, (0, 16), (w - 1, h - 1), color, 2)
    cv2.putText(t, f"{row.short} {row.fc75_score_new:.2f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (20, 20, 20), 1,
                cv2.LINE_AA)
    return t


def sheet(rows, root, out, cols, scale, title):
    w, h = int((22 / SCALE + 2 * MARGIN) * scale) + 4, int((66 / SCALE + 2 * MARGIN) * scale) + 20
    tiles = [tile(cv2.imread(str(root / r.image)), r, w, h, scale) for r in rows.itertuples()]
    n_rows = (len(tiles) + cols - 1) // cols
    canvas = np.full((n_rows * h + 30, cols * w, 3), 255, np.uint8)
    cv2.putText(canvas, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        canvas[30 + r * h:30 + (r + 1) * h, c * w:(c + 1) * w] = t
    cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    root, labels, out = Path(sys.argv[1]), pd.read_csv(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    tpl = cv2.imread(str(Path(__file__).resolve().parent / "fc75.png"), cv2.IMREAD_GRAYSCALE)
    tpls = rotations(cv2.resize(tpl, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA))
    recs = []
    for r in labels.itertuples():
        g = cv2.resize(cv2.imread(str(root / r.image), cv2.IMREAD_GRAYSCALE), None, fx=SCALE, fy=SCALE,
                       interpolation=cv2.INTER_AREA)
        s, (x, y), (th, tw) = match(g, tpls)
        recs.append({"image": r.image, "fc75_score_new": round(float(s), 4), "x": x, "y": y, "tw": tw, "th": th})
    df = labels.merge(pd.DataFrame(recs), on="image")
    df["short"] = df.image.map(lambda p: p.split("/")[-2][0] + p.split("/")[-1].split(".")[0])
    df["defect"] = df.label != "normal"
    diff = float((df.fc75_score - df.fc75_score_new).abs().max())
    df.drop(columns=["short"]).to_csv(out / "fc75_match_all.csv", index=False)

    # 1. 점수 분포
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bins = np.linspace(0.3, 1.0, 71)
    ax.hist(df[~df.defect].fc75_score_new, bins=bins, alpha=0.7, label=f"normal (n={int((~df.defect).sum())})")
    ax.hist(df[df.defect].fc75_score_new, bins=bins, alpha=0.7, label=f"defect (n={int(df.defect.sum())})")
    ax.axvline(THR, color="k", ls="--", lw=1, label="threshold 0.7")
    ax.set(xlabel="FC-75 template match score (max over ±10° rotations)", ylabel="images", yscale="log",
           title="FC-75 print match score of all 1,105 pcb4 images")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "fc75_score_hist.png", dpi=130)
    plt.close()

    # 2. FC-75로 라벨된 190장, 3. 경계에 가장 가까운 비FC-75 100장, 4. 전체 1,105장
    pos = df[df.fc75].sort_values("fc75_score_new")
    neg = df[~df.fc75].sort_values("fc75_score_new", ascending=False)
    sheet(pos, root, out / "fc75_positive_all.jpg", 19, 0.62,
          f"All {len(pos)} images labelled FC-75 (green; blue = decided visually), lowest score first")
    sheet(neg.head(100), root, out / "fc75_negative_top100.jpg", 20, 0.62,
          "100 non-FC-75 images with the highest scores (closest to the threshold): gray border")
    sheet(df.sort_values("fc75_score_new", ascending=False), root, out / "fc75_all_sorted.jpg", 34, 0.42,
          "All 1,105 images sorted by score (green = FC-75 label, gray = non-FC-75, blue = decided visually)")

    # 5. 눈으로 판정한 6장: 인쇄가 결함에 가린 경우가 있어 기판 전체를 보여 준다
    from contact_sheet import board_crop
    notes = {"A018": "HW-168 board (no FC-75 print) -> non-FC-75",
             "A035": "defect covers the print -> FC-75 (chip 4056E, layout)",
             "A039": "red sticker covers the print -> FC-75",
             "A079": "scratch over the print, still readable -> FC-75",
             "A093": "print visible next to defect -> FC-75",
             "A094": "print visible next to defect -> FC-75"}
    vis = df[df.source == "visual"]
    tw, th = 420, 350
    canvas = np.full((2 * (th + 40) + 30, 3 * tw, 3), 255, np.uint8)
    cv2.putText(canvas, "6 images decided by eye (template score unreliable or near threshold)", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    for k, r in enumerate(vis.itertuples()):
        c = board_crop(cv2.imread(str(root / r.image)))
        sc = min(tw / c.shape[1], th / c.shape[0])
        c = cv2.resize(c, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        y0, x0 = 30 + (k // 3) * (th + 40), (k % 3) * tw
        canvas[y0 + 34:y0 + 34 + c.shape[0], x0:x0 + c.shape[1]] = c
        cv2.putText(canvas, f"{r.short} score {r.fc75_score_new:.2f}", (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, notes.get(r.short, ""), (x0 + 4, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 120, 0), 1,
                    cv2.LINE_AA)
    cv2.imwrite(str(out / "fc75_visual_decisions.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])

    gap = df[(df.fc75_score_new > 0.6) & (df.fc75_score_new < 0.7)]
    summary = {
        "recomputed_vs_csv_max_abs_diff": diff,
        "fc75_defect": int((df.fc75 & df.defect).sum()), "defect": int(df.defect.sum()),
        "fc75_normal": int((df.fc75 & ~df.defect).sum()), "normal": int((~df.defect).sum()),
        "images_in_0.6_0.7": gap.short.tolist(),
        "visual_rows": df[df.source == "visual"][["short", "fc75", "fc75_score_new"]].values.tolist(),
        "max_score_non_fc_normal": float(df[~df.fc75 & ~df.defect].fc75_score_new.max()),
        "min_score_fc_normal": float(df[df.fc75 & ~df.defect].fc75_score_new.min()),
    }
    print(summary)


if __name__ == "__main__":
    main()
