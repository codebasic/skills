---
name: timeseries
description: '시계열 예측·분류·이상탐지 전 과정을 안내하는 스킬. Jena 기후 예측, HAR 센서 활동분류, AML 불균형 이상탐지, Seismic 지진파형 생성 실습 기반. 슬라이딩 윈도우, persistence 베이스라인, MLP/Conv1D/LSTM/TCN/Transformer 모델링, Saliency 설명력 분석, TimesFM/TTM/MOMENT 시계열 파운데이션 모델 제로샷·파인튜닝 및 언어 모델 시계열 직렬화까지 다룹니다. Use when: 시계열 전처리, 리샘플링, 시계열 트랜스포머, 시계열 분류, 시계열 회귀, 시계열 설명력, Saliency, MOMENT, TimesFM, TTM, 시계열 파운데이션 모델, 불균형 이벤트 시계열.'
argument-hint: '시계열 예측·분류 전처리와 트랜스포머·파운데이션 모델링'
user-invocable: true
---

# 시계열 예측·분류·이상탐지 (전처리 → 딥러닝/트랜스포머 → 파운데이션 모델)

시간 순서가 있는 수치 데이터와 이벤트 시퀀스로 "미래 시점 예측(회귀)", "다채널 상태 분류(Classification)", "불균형 이상 탐지(Anomaly Detection)", "파형 생성"을 수행하는 전 과정의 실전 경험과 교훈을 정리한 스킬입니다.
Jena 기후 예측 실습, HAR 3축 센서 활동인식, AML 불균형 거래 탐지, Seismic 지진파형 연구의 실제 벤치마크와 함정을 바탕으로 작성되었습니다.

## When to Use

- 시계열 CSV를 적재하고 정제·리샘플링할 때
- 슬라이딩 윈도우 시퀀스 데이터셋을 만들 때 (`timeseries_dataset_from_array`)
- 시계열 회귀·분류 모델(MLP / 1D CNN / LSTM / TCN / Transformer)을 구축·비교할 때
- persistence 베이스라인 대비 성능을 평가할 때
- 시계열 트랜스포머(Time-Series Transformer) 인코더를 구현하고 위치 임베딩을 구성할 때
- Keras(PyTorch 백엔드)에서 Saliency / Gradient × Input으로 시점·채널 기여도를 분석(XAI)할 때
- TimesFM, TTM, MOMENT 등 시계열 파운데이션 모델(FM)을 제로샷/파인튜닝으로 활용할 때
- 불균형 이벤트/트랜잭션 시계열을 자연어 텍스트 문맥으로 직렬화(Prompting)하여 사전훈련 LM을 활용할 때
- WSL / 바인드 마운트 컨테이너 환경에서 PyTorch 버스 에러를 방지하고 가상환경을 격리할 때

## 참고 자료

| 파일 | 내용 |
|---|---|
| `references/preprocessing.md` | 적재·정제·리샘플링·분할·표준화 |
| `references/sequence-datasets.md` | 슬라이딩 윈도우 시퀀스 생성 (제너레이터 주의점) |
| `references/models.md` | 베이스라인, MLP/Conv1D/LSTM/TCN/Transformer 모델링, HAR 센서 벤치마크, Saliency 설명력, Seismic 주파수-시간 변환 |
| `references/foundation-models.md` | TimesFM/TTM/MOMENT/LM 직렬화 가이드, 실측 비교, momentfm 결함 해결, venv 격리 모범 사례 |
| `scripts/timeseries_forecasting.py` | 전체 파이프라인 참조 구현 (conda pytorch 환경) |

## 핵심 교훈 (요약)

1. **전처리**: 인덱스 정렬 단조성 검증 → 센서 오류값(-9999) 교체 → 리샘플링 → `shuffle=False` 분할 → 훈련 데이터로만 fit한 표준화. 전역 인덱싱이 필요하면 **전체 구간을 한 번에 표준화한 배열**을 별도로 유지.
2. **시퀀스 제너레이터는 1회성**: 순회하면 소진됨. 베이스라인 계산 등에 쓴 뒤 모델 학습용으로 다시 생성해야 함.
3. **persistence 베이스라인이 매우 강력**: 24시간 뒤 예측은 "마지막 값 복사"만으로 MAE ≈ 2.4~2.6°C. 어떤 모델이든 이 기준선을 먼저 계산하고 비교해야 함.
4. **모델 선택 실측**: `Conv1D + GlobalAveragePooling1D`는 회귀에서 시간축 정보를 파괴해 최악. 2층 LSTM은 1층보다 과적합. **다중 출력(온도+습도 동시 예측)이 단일 출력보다 낫고 유일하게 베이스라인을 상회** — 공변량 활용 + 정규화 효과.
5. **학습 설정**: rmsprop 1e-3 + patience 3은 val_loss 요동을 유발. 학습률을 낮추고 스케줄러를 쓸 것. GPU 가동률은 배치 512로 확대해 개선.
6. **고빈도 센서 분류에서는 1D CNN/TCN이 트랜스포머 압도**: 국소 파형 모티프가 핵심인 HAR 3축 센서 분류에서는 1D CNN(99.03%)과 TCN(95.32%)이 MOMENT 파운데이션 모델(83.11%)보다 정확도와 학습 속도 면에서 월등함. 무조건 트랜스포머를 고집하기보다 신호의 물리적 특성(진폭, 국소 주기)을 우선 고려할 것.
7. **MOMENT-1-small 함정**: `momentfm 0.1.4`는 `from_pretrained(..., model_kwargs={"task_name": "embedding"})`로 생성해도 내부 `task_name`이 `reconstruction`으로 남아 `out.embeddings`가 None이 됨. 로드 직후 `model.task_name = "embedding"`을 명시적으로 재할당해야 함. 입력 텐서는 반드시 `(B, C, T)`로 permute 필요. 인스턴스 정규화로 인해 신호의 절대 진폭 정보가 지워지는 점 주의.
8. **이벤트 시계열의 LM 직렬화(Serialization)**: 비정형 메타데이터(금액, 국가, 경과시간 등)가 결합된 불균형 거래 시계열(AML)에서는 이벤트 로그를 텍스트 시퀀스로 직렬화하여 사전훈련 언어 모델(ELECTRA)을 활용하는 것이 일반 테이블/시계열 트랜스포머보다 PR-AUC(0.428) 및 F1(0.436) 면에서 월등함.
9. **시계열 설명력 (Saliency Map)**: Keras(torch 백엔드)에서는 `torch.autograd.grad`를 사용하여 입력 텐서 대비 그래디언트($\text{Input} \times \text{Gradient}$)를 즉시 추출 가능. imshow((채널, 시간)) 히트맵으로 시각화하여 결정적 시점과 채널 기여도를 해석.
10. **환경 격리 및 버스 에러 방지**: Windows 바인드 마운트(`/workspace`)에서 venv를 구성하고 PyTorch를 로드하면 SIGBUS 버스 에러가 발생함. 독립 venv(MOMENT, CGM-GM 등)는 반드시 로컬 파일시스템(`/home/me/venvs/`)에 생성하고, 패키지 충돌 방지를 위해 태스크별로 분리 실행할 것.

## 실측 벤치마크 요약

### 1. 장기 기후 회귀 (24시간 뒤 온도 MAE, °C)
- **LSTM 다중출력 (온도+습도)**: **2.351** (유일하게 베이스라인 상회)
- **Persistence 베이스라인**: 2.426
- **TimesFM 파인튜닝 (헤드만)**: 2.461
- **LSTM (1층)**: 2.455 / **TimesFM 제로샷**: 2.517 / **TTM 제로샷**: 2.586 / **MLP**: 2.591 / **Conv1D**: 3.314

### 2. 센서 활동 분류 (HAR 3축 가속도계, 6개 클래스)
- **1D CNN**: **Accuracy 99.03%** / F1 0.985 (학습 ~30초, 최적)
- **CNN + LSTM**: Accuracy 98.50% / F1 0.981
- **TCN**: Accuracy 95.32% / F1 0.943
- **BiLSTM**: Accuracy 95.70% / F1 0.952
- **MOMENT-1-small (동결 FM + 헤드)**: Accuracy 83.11% / F1 0.688 (도메인 괴리 & 진폭 손실)
- **MLP (요약 통계)**: Accuracy 62.00% / F1 0.580

### 3. 극단적 불균형 이벤트 이상 탐지 (AML, 사기율 ~0.2%)
- **Pretrained Transformer (ELECTRA 텍스트 직렬화)**: **PR-AUC 0.4285** / **Fraud F1 0.4359**
- **MLP (테이블 특징)**: PR-AUC 0.4512 / Fraud F1 0.1906 (정밀도 불균형)
- **FT-Transformer**: PR-AUC 0.1442 / Fraud F1 0.0106
- **Baseline (무작위 추측)**: PR-AUC 0.0021
