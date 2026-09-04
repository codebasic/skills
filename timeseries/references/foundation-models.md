# 시계열 파운데이션 모델 및 트랜스포머: TimesFM / TTM / MOMENT / LM 직렬화

사전 훈련된 시계열 파운데이션 모델(FM)과 대규모 사전훈련 트랜스포머는 **제로샷 예측, 특징 임베딩, 도메인 전이**를 제공합니다.
다만 도메인 특성, 시계열 주파수, 공변량 유무에 따라 실측 성능과 유효성이 크게 달라집니다.
Jena 기후 회귀, HAR 센서 활동 분류, AML 이상 거래 탐지 프로젝트에서 검증된 핵심 경험과 함정을 정리합니다.

## 1. 모델 선택 가이드

| 모델 / 기법 | 입력 형태 | 주요 태스크 | 사용 경로 / 백본 | 비고 |
|---|---|---|---|---|
| **TimesFM 2.0** (Google) | ❌ 단변량 `(B, T)` | 시계열 예측(회귀) | `transformers` (4.56/4.57) | 패치 기반 디코더, 지평 128 고정 |
| **TTM** (IBM) | ✅ 다변량 `(B, T, C)` | 시계열 예측(회귀) | `granite-tsfm` 패키지 | 80만 파라미터 경량 TSMixer, 채널 독립/믹싱 |
| **MOMENT-1-small** (AutonLab) | ✅ 다변량 `(B, C, T)` | 임베딩·분류·예측 | `momentfm` (`~/venvs/moment`) | 범용 시계열 트랜스포머, 인스턴스 정규화 내장 |
| **LM 시계열 직렬화** (NLP FM) | ✅ 텍스트 시퀀스 | 불균형 이벤트·이상탐지 | `transformers` (ELECTRA 등) | 시계열 메타데이터/경과시간을 텍스트 문맥으로 직렬화 |
| Moirai (Salesforce) | ✅ any-variate | 시계열 예측 | `uni2ts` 별도 패키지 | transformers 4.57에서 제거됨 |
| Chronos-Bolt (Amazon) | ❌ 단변량 | 시계열 예측 | `chronos-forecasting` | transformers 4.57에서 제거됨 |

## 2. ⚠️ 버전 호환성 함정 (중요)

- **TimesFM**: `google/timesfm-2.0-500m-pytorch`는 transformers **4.56/4.57**에서
  정상 로드·동작(사인파 검증 MAE 0.146). `timesfm-2.5-200m-pytorch`는 구현과
  가중치 구조가 달라 **전부 MISSING/UNEXPECTED로 로드되어 무작위 가중치처럼
  동작**합니다 — 로딩 보고서에 MISSING이 보이면 결과를 믿지 마세요.
- **Moirai/TTM/Chronos는 transformers 4.57에서 제거**되었습니다.
  TTM은 `pip install granite-tsfm` 사용(설치 시 transformers가 4.56으로 조정됨).
- 커널이 구버전 transformers를 들고 있는 상태에서 granite-tsfm을 설치하면
  `tsfm_public` 패키지 초기화(Trainer 체인 임포트)가 버전 충돌로 실패합니다.
  이때는 **패키지 초기화를 건너뛰고 TTM 모델 모듈만 직접 로드**합니다:

```python
import importlib, importlib.util, sys, types

try:
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
except ImportError:
    _spec = importlib.util.find_spec('tsfm_public')  # 실행 없이 위치만 탐색
    _pkg = types.ModuleType('tsfm_public')
    _pkg.__path__ = list(_spec.submodule_search_locations)
    sys.modules['tsfm_public'] = _pkg
    _sub = types.ModuleType('tsfm_public.models')
    _sub.__path__ = [list(_spec.submodule_search_locations)[0] + '/models']
    sys.modules['tsfm_public.models'] = _sub
    _ttm = importlib.import_module('tsfm_public.models.tinytimemixer')
    TinyTimeMixerForPrediction = _ttm.TinyTimeMixerForPrediction
```

## 3. TimesFM (단변량, 5억 파라미터)

```python
from transformers import TimesFmModelForPrediction

timesfm = TimesFmModelForPrediction.from_pretrained(
    'google/timesfm-2.0-500m-pytorch', torch_dtype=torch.float32).eval().to(device)

# 제로샷 예측: 과거 512시점 -> mean_predictions (배치, 128)
with torch.no_grad():
    out = timesfm(past_values=[x_512], freq=[0], forecast_context_len=512)
예측 = out.mean_predictions[:, 23]  # 24시간 뒤 = 지평 인덱스 23
```

- `freq` 지시자(0=고빈도, 1=중간, 2=저빈도)와 컨텍스트 길이(512~2048)를
  실험한 결과 **freq=0 + 512가 최선**(2.679), 컨텍스트 확장 효과는 0.008 수준.
- 배치 입력: `past_values=[x1, x2, ...]`, `freq=[0]*B` 리스트로 전달.
- 파인튜닝 입력 `future_values`는 (B, 128) — MSE + 분위수 손실이 자동 계산됨.

### 헤드만 파인튜닝 (디코더 동결)

```python
for p in timesfm.parameters():
    p.requires_grad_(False)
for p in timesfm.horizon_ff_layer.parameters():   # 출력 헤드(약 492만 파라미터)
    p.requires_grad_(True)

optimizer = torch.optim.AdamW(timesfm.horizon_ff_layer.parameters(), lr=3e-4)
# 학습: timesfm(past_values=xs, freq=[0]*B, future_values=targets).loss.backward()
# 매 에포크 검증 MAE로 최적 헤드만 deepcopy해 보관 후 복원
```

결과: val 2.517 → 2.461 (개선 +0.06, 베이스라인에는 미달).
**원인: TimesFM은 단변량이라 14개 공변량을 못 씀.** 헤드만 학습해도
디코더가 온도의 국지 역학을 새로 배우지 못합니다.

## 4. TTM (다변량, 80만 파라미터)

```python
ttm = TinyTimeMixerForPrediction.from_pretrained(
    'ibm-granite/granite-timeseries-ttm-r2', revision='main',  # 512 -> 96
    num_input_channels=14).eval()

# (B, 512, 14) 다변량 입력 -> (B, 96, 14) 다변량 예측
out = ttm(past_values=x)                     # x: (B, 512, C) float32
preds = out.prediction_outputs[:, 23, 온도열위치]  # 24시간 뒤, 온도 채널
```

- **반드시 3차원** `(batch, seq_len, channels)` 입력. `seq_len == config.context_length`.
- 사인파 검증 MAE 0.025로 매우 정확 — 모델 자체는 정상.
- 그러나 jena 결과는 **val 2.586 / test 2.816로 단변량 TimesFM보다 나쁨**.
  이유: 제로샷 체크포인트는 **채널 독립 모드**(채널 간 상호작용 없음)라
  14개 채널을 넣어도 실질적으로 독립된 단변량 예측의 나열이며,
  온도 전용 대규모 사전훈련을 받은 TimesFM이 오히려 유리했습니다.
- 공변량 상호작용을 살리려면 **채널 믹싱 모드로 파인튜닝**이 필요합니다.

---

## 5. MOMENT-1-small (범용 시계열 파운데이션 모델)

CMU & AutonLab에서 개발한 시계열 멀티태스크 파운데이션 모델로, T5 기반 트랜스포머 아키텍처에 시계열 패치 인코딩을 결합한 모델입니다.

### 5.1 ⚠️ momentfm 0.1.4 핵심 결함 및 해결책

`momentfm` 0.1.4 버전에서는 `from_pretrained` 호출 시 `task_name="embedding"` 파라미터를 넘기더라도 내부 속성 `model.task_name`이 config의 기본값(`"reconstruction"`)으로 유지되는 버그가 있습니다. 이 경우 `out.embeddings`가 `None`으로 반환되어 하류 분류 작업이 불가능해집니다.

```python
from momentfm import MOMENTPipeline

def load_moment(model_id="AutonLab/MOMENT-1-small", device="cuda"):
    model = MOMENTPipeline.from_pretrained(
        model_id,
        model_kwargs={"task_name": "embedding"},
    ).to(device).eval()
    
    # ⚠️ 필수: 버그 우회를 위해 직접 task_name 속성을 재할당해야 함!
    model.task_name = "embedding"
    
    # 백본 가중치 동결
    for p in model.parameters():
        p.requires_grad_(False)
    return model
```

### 5.2 텐서 차원 및 임베딩 추출 패턴

MOMENT는 표준 시계열 `(batch, seq_len, channels)` 대신 **`(batch, channels, seq_len)`** 형태를 기대합니다. 따라서 반드시 차원 전치(`permute(0, 2, 1)`)를 거쳐야 합니다.

```python
@torch.no_grad()
def extract_moment_embeddings(moment_model, X: np.ndarray, batch_size: int = 256, device="cuda") -> np.ndarray:
    """
    X: (N, T, C) 넘파이 시계열 배열
    반환: (N, d_model) 임베딩 벡터 배열
    """
    feats = []
    for i in range(0, len(X), batch_size):
        # (B, T, C) -> (B, C, T) 변환
        xb = torch.from_numpy(X[i:i + batch_size]).permute(0, 2, 1).float().to(device)
        mask = torch.ones_like(xb[:, 0, :])  # 패딩 없는 경우 전부 1
        out = moment_model(x_enc=xb, input_mask=mask)
        feats.append(out.embeddings.float().cpu())
    return torch.cat(feats, dim=0).numpy()
```

### 5.3 실측 결과 및 도메인 괴리 (HAR 센서 프로젝트)

동결된 MOMENT 백본 위에 경량 MLP 분류 헤드를 올려 평가한 결과:
- **실측 성능**: Accuracy **83.11%**, Macro F1 **0.688** (지도학습 1D CNN의 99.03% 대비 현저히 낮음)
- **원인 분석**:
  1. **인스턴스 정규화(Instance Normalization)**: MOMENT 내부는 입력 시계열의 평균/분산을 0/1로 정규화하여 처리합니다. 그러나 센서 활동 인식(정지 상태 vs 전력 질주)에서는 **신호의 절대 진폭(진동 세기, 중력 가속도 크기)**이 결정적인 구분 단서인데, 이 절대 크기 정보가 지워집니다.
  2. **사전훈련 데이터 분포 차이**: 고빈도 3축 생체역학 센서 데이터는 대규모 일반 시계열 코퍼스(기상, 전력, 주가 등)와 파형 특성이 크게 상이합니다.
- **교훈**: 물리 센서/활동 인식과 같이 신호 진폭과 국소 주기가 핵심인 분야에서는 범용 시계열 FM 동결 추출보다 **도메인 특화 1D CNN / TCN 지도학습**이 훨씬 우수합니다.

---

## 6. 이벤트/트랜잭션 시계열의 언어 모델 직렬화 (Serialization/Prompting)

불규칙하고 다양한 메타데이터(금액, 범주, 발신/수신 주체, 경과시간 등)가 결합된 트랜잭션 시계열 이상 탐지(AML 프로젝트)에서는, 전통적 테이블 모델이나 수치 트랜스포머보다 **사전훈련 언어 모델(ELECTRA/BERT)을 이용한 텍스트 직렬화**가 매우 강력한 성능을 발휘합니다.

### 6.1 시계열 이벤트 문맥 직렬화

시계열의 이벤트 발생 순서와 경과시간, 노드 메타 정보를 자연어 템플릿으로 구조화합니다.

```python
def serialize_transaction_event(row, src_node, dest_node):
    return (
        f"Transaction type {row['type']} of {row['amount']:.2f} {row['currency']} at {row['timestamp']}. "
        f"Origin: {src_node['type']} in {src_node['country']} (risk: {src_node['is_fraud']}). "
        f"Destination: {dest_node['type']} in {dest_node['country']} (risk: {dest_node['is_fraud']}). "
        f"Elapsed time since last event: {row['time_since_previous_transaction']}."
    )
```

### 6.2 불균형 시계열 이상 탐지 실측 비교 (사기율 ~0.2%)

| 모델 | PR-AUC (Average Precision) | Fraud F1-Score | 비고 |
|---|---|---|---|
| **Pretrained Transformer (ELECTRA-small)** | **0.4285** | **0.4359** | 텍스트 직렬화 + 분류 헤드 선형 프로빙 |
| MLP (테이블 수치/원핫 인코딩) | 0.4512 | 0.1906 | PR-AUC는 높으나 F1 불균형 심각 (정밀도 0.10) |
| FT-Transformer (Feature Tokenizer) | 0.1442 | 0.0106 | 희소 불균형 범주 간 상호작용 학습에 과적합 |
| Baseline (Random Guess) | 0.0021 | - | 데이터셋 기본 양성 비율 |

- **장점**: 사전훈련 언어 모델은 대규모 코퍼스로부터 획득한 상식적 맥락(예: 국가 간 송금 리스크, 짧은 경과 시간 동안의 급격한 금액 이동 등)을 바탕으로 극단적 불균형 상황에서도 높은 정밀도-재현율 균형을 유지합니다.

---

## 7. 실행 환경 및 격리 모범 사례 (WSL / Docker / 가상환경)

시계열 파운데이션 모델(MOMENT, TimesFM, TTM)과 다양한 프레임워크를 병용할 때 발생하는 실전 환경 오류 방지 원칙입니다.

1. **바인드 마운트 버스 에러 (SIGBUS / Bus Error 방지)**:
   - Windows 호스트와 공유되는 바인드 마운트 볼륨(`/workspace`) 위에 venv를 생성하고 `import torch`를 수행하면 가상 메모리 매핑 불일치로 버스 에러가 발생할 수 있습니다.
   - **원칙**: 모든 독립 venv(MOMENT용, cgm-gm용 등)는 **반드시 컨테이너 로컬 파일시스템(`/home/me/venvs/`)**에 `uv venv`로 생성하십시오.
2. **패키지 버전 충돌 격리**:
   - `granite-tsfm`은 `transformers==4.56.x`를 요구하고, 최신 LLM 도구는 `transformers>=4.57`을 요구합니다.
   - 워크스페이스 표준인 `conda pytorch` 환경을 오염시키지 말고, 특수 모델(MOMENT, 지진 CGM-GM 등)은 전용 로컬 venv의 파이썬 인터프리터(`~/venvs/moment/bin/python`)로 격리 실행하십시오.

---

## 8. 평가 프로토콜 (일관성 유지)

- 동일한 윈도우 시점, 동일한 평가 척도(회귀는 원래 단위 MAE, 불균형 분류는 PR-AUC & Macro F1)로 비교
- 배치 추론으로 가속 (64~256 단위)
- 전역 인덱싱 시 전체 구간 표준화 배열 사용 (분할별 로컬 배열과 혼동 금지)
- **로그/변수 재사용**: 재실행 방지를 위해 `결과 = 결과 if '결과' in globals() else {}` 패턴

## 9. 종합 결론 및 의사결정 매트릭스

| 문제 유형 | 추천 모델 아키텍처 | 이유 및 실측 근거 |
|---|---|---|
| **고빈도 센서 / 파형 분류 (HAR, 진동)** | **1D CNN / TCN** | 주기적 국소 형상이 핵심. 1D CNN(99.0%)이 트랜스포머(83.1%)를 압도 |
| **물리 파형 생성 / 시계열 역학 (지진파)** | **STFT/FAS 주파수 변환 + 잠재 모델** | 시간 도메인 직접 어텐션의 모드 붕괴 방지 및 물리적 포락선 학습 용이 |
| **장기 단변량 기후/수요 예측 (Jena)** | **다중출력 LSTM 또는 TimesFM 파인튜닝** | 제로샷 FM은 persistence 베이스라인에 미달. 공변량 동시 예측이 최고 성능 |
| **비정형 메타데이터 결합 이벤트 시계열 (AML)** | **사전훈련 언어 모델 직렬화 (LM Prompting)** | 불균형 이상 거래에서 정밀도-재현율 균형 우수 (Fraud F1 0.436) |
