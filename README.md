# 프로젝트 6. 제조업 실시간 이상 탐지 시스템 구축

## 개요
- 목표: VisA pcb4 데이터로 정상 이미지만 학습하는 이상 탐지 모델(PaDiM, AutoEncoder, One-Class SVM)을 비교하고, PaDiM을 Gradio로 실시간 추론한다. 이어서 PaDiM의 높은 점수가 실제로 결함을 보고 나온 것인지 검증한다.
- 모델: PaDiM (Wide-ResNet50), AutoEncoder, One-Class SVM, 비교용 CatBoost
- 데이터: [VisA](https://github.com/amazon-science/spot-diff) pcb4 (정상 1,005장, 불량 100장). 노트북이 공식 주소에서 받는다.
- 실행 환경: Google Colab T4 GPU

## 파일 구성
| 파일 | 내용 |
|---|---|
| [`06_interpreter.ipynb`](06_interpreter.ipynb) | 실습 코드 + 정리 + Test 1~4 분석 + 개인 회고 |
| `pcb4_variant_labels.csv` | pcb4 1,105장의 기판 종류(`FC-75` 인쇄 여부) 라벨. Test 1~4에서 사용 |
| `scripts/` | 기판 종류 라벨을 만든 템플릿 매칭 코드(`variant_score.py`, `textmask_check.py`)와 템플릿(`fc75.png`), 검증 그림을 만드는 코드(`evidence_sheets.py`, `contact_sheet.py`) |
| `evidence/` | 기판 종류 라벨 검증 그림과 점수 (점수 분포, 인쇄 부분 격자 이미지) |
| `requirements.txt` | 필요한 패키지 |

## 노트북 구성
1. 라이브러리 설치 · VisA 데이터셋 다운로드
2. Dataloader 구현 · 시각화
3. 모델 구현 · 비교 평가 (PaDiM, AutoEncoder, One-Class SVM)
4. 실시간 inference (Gradio)
5. 추가 실험
   - Test 1: PaDiM이 결함이 아니라 기판 종류를 보고 맞혔는지 검증
   - Test 2: 점수가 결함 영역에서 나오는지(실험 A), 이상 맵이 결함 위치를 가리키는지(실험 B) 검증
   - Test 3: AutoEncoder의 AUROC가 무작위보다 낮은 원인을 epoch별 학습 기록으로 확인
   - Test 4: 불량 라벨을 쓰는 CatBoost를 Test 1과 같은 잣대로 비교
6. 개인 회고 (KPT, AAR)

## 실행 방법
1. Colab에서 노트북 열기 (런타임: T4 GPU)
2. `pcb4_variant_labels.csv`를 `/content`에 올리기
3. 위에서부터 차례로 실행

## 회고
### KPT
- Keep: AutoEncoder에서 점수가 생각보다 낮았던 것을 의심해 문제를 찾고 돌린 방식이나 데이터셋 기판에 몇가지 다른 점을 보고 혹시나 해서 돌린 실험처럼, 앞으로도 여러 방면으로 미리 이런 벤치마크 테스트를 돌리고, 비정상일 때 최대한 여러 방면으로 자세하게 확인해 볼 것이다.
- Problem: AutoEncoder가 0.27로 무작위보다 낮았는데, 모델 문제가 아니라 입력과 출력의 범위가 안 맞는 코드 문제였고, 기판 종류 라벨을 만들 때 결함이 인쇄를 가린 사진 때문에 자동 판별이 틀려 시간이 더 들었다.
- Try: 새 모델을 쓸 때는 입력·출력 범위와 손실이 제대로 줄어드는지부터 확인하고, 데이터셋은 미리 자세하게 확인해 볼 것이다.

### AAR
- 목표: 정상 이미지로만 학습하는 이상 탐지 모델 3개를 비교해 보고, 가장 잘 나온 PaDiM이 정말 불량을 보고 맞힌 건지 확인해 본다.
- 결과: PaDiM은 같은 기판끼리 비교해도 불량을 잘 찾아서 정말 불량을 보고 있었지만, FC-75 기판의 정상 제품은 불량으로 잘못 보는 경우가 2배 넘게 많았다. 또 AutoEncoder는 코드를 고치니 크게 좋아졌고, CatBoost는 점수는 제일 높았지만 기판 종류로 맞히는 경향이 있었다.
- 차이 원인: 일단 학습에 쓴 정상 사진에 FC-75 기판이 거의 없었고, AutoEncoder는 입력과 출력의 값 범위가 서로 달라서 학습이 잘 되지 않았었다. CatBoost는 불량 사진이 거의 다 FC-75 기판이라 기판 종류까지 같이 배웠다.
- 다음 액션: 학습 데이터에 기판 종류를 골고루 넣어서 다시 학습해 보거나 기판마다 기준을 따로 정해 보고, 실험할 때 시드를 고정해서 결과를 다시 확인해 볼 것이다.
