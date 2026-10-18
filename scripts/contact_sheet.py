"""기판 영역을 잘라 격자 이미지로 만든다 (육안 검증용).

usage: contact_sheet.py <VisA root> <scores csv> <out png> <image rel path>...
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TW, TH, COLS = 300, 250, 5


def board_crop(img):
    # 파란 기판 영역(HSV) 중 가장 큰 덩어리의 bbox
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (90, 80, 40), (130, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    pad = 20
    return img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]


def main():
    root, scores, out = Path(sys.argv[1]), pd.read_csv(sys.argv[2]), sys.argv[3]
    by_img = scores.set_index("image")
    tiles = []
    for rel in sys.argv[4:]:
        img = cv2.imread(str(root / rel))
        c = board_crop(img)
        s = min(TW / c.shape[1], (TH - 30) / c.shape[0])
        c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        tile = np.zeros((TH, TW, 3), np.uint8)
        tile[30:30 + c.shape[0], :c.shape[1]] = c
        name = rel.split("/")[-2][0] + rel.split("/")[-1].split(".")[0]  # N0001 / A039
        txt = f"{name} {by_img.loc[rel, 'fc75_score']:.2f}"
        cv2.putText(tile, txt, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) % COLS:
        tiles.append(np.zeros((TH, TW, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
    cv2.imwrite(out, np.vstack(rows))
    print("saved", out, len(sys.argv) - 4)


if __name__ == "__main__":
    main()
