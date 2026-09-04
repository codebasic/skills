# 모델링: 베이스라인 → MLP / Conv1D / LSTM → 비교 평가

## 0. 베이스라인을 먼저 계산하라

가장 단순한 가정 — "지금 값이 24시간 뒤와 같다"(persistence) — 가
시계열 회귀에서는 의외로 강력합니다. 어떤 모델도 이 기준선을 넘지 못하면
모델이 아무것도 배우지 못한 것입니다. **모델링 전에 반드시 계산**합니다.

```python
baseline_mae = {}
for split in seq_dataset:
    samples = list(seq_dataset[split])
    마지막행 = np.array([x[-1] for x, y in samples])          # (N, 14) 마지막 시점
    예측 = scaler.inverse_transform(마지막행)[:, 온도열위치]   # 원래 단위 복원
    목표 = np.array([y for x, y in samples])
    baseline_mae[split] = np.mean(np.abs(예측 - 목표))
# 결과: val 2.426 / test 2.606 °C
```

## 1. 모델 구축 컨벤션

모델 구축은 `build_모델이름()` 형태의 함수로 추출하고, 회귀 출력층에는
활성화 함수를 넣지 않습니다.

```python
def build_mlp():
    """은닉층 하나(유닛 16개)를 가진 MLP 모델."""
    inputs = keras.Input(shape=(시퀀스길이, 특성수))
    x = keras.layers.Flatten()(inputs)          # (120, 14) -> (1680,)
    x = keras.layers.Dense(16, activation='relu')(x)
    outputs = keras.layers.Dense(1)(x)          # 회귀: 활성화 없음
    return keras.Model(inputs, outputs, name='mlp')
```

## 2. 학습: 체크포인트 + 조기 종료

```python
체크포인트폴더 = Path('./checkpoints')
체크포인트폴더.mkdir(exist_ok=True)

history = model.fit(
    dataloader['train'],
    validation_data=dataloader['val'],
    epochs=30,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            filepath=체크포인트폴더 / 'lstm_rnn.keras',
            monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ],
)
```

> ⚠️ **평가 시점에서는 저장된 체크포인트를 load**하세요. 학습 중 `clear_session`
> 후 재구축된 인메모리 모델은 컴파일되지 않아 `evaluate` 시 오류가 납니다.
> ```python
> best_rnn = keras.models.load_model(체크포인트폴더 / 'lstm_rnn.keras')
> best_rnn.evaluate(dataloader[split], return_dict=True)['mae']
> ```

## 3. 아키텍처별 실측 교훈 (24시간 뒤 온도 예측, MAE °C)

| 아키텍처 | val | test | 교훈 |
|---|---|---|---|
| MLP (Flatten + Dense 16) | 2.591 | 2.770 | 5일치를 1차원으로 펼쳐 최근 값에 과의존 |
| **Conv1D + GlobalAvgPooling** | 3.314 | 3.562 | ❌ **최악** — 풀링·평균화가 시간축 위상 정보를 파괴 |
| LSTM (1층, 16유닛) | 2.455 | 2.739 | 시간 구조 활용으로 양호, 학습 느림(285ms/step) |
| LSTM (2층, return_sequences) | 2.458 | 2.657 | 학습 MAE는 낮으나 과적합 + 학습 시간 2배. 이 규모에선 1층이 효율적 |
| **LSTM 다중출력 (온도+습도)** | **2.351** | **2.498** | ✅ **유일하게 베이스라인 상회** — 공변량 공동 학습 + 정규화 효과 |

### 핵심 교훈

- **Conv1D 뒤 `GlobalAveragePooling1D` 금지**: "어제 같은 시각" 같은 지연 위상이
  평균으로 뭉개집니다. 시계열 회귀에서는 마지막 시점 상태를 쓰거나
  `Flatten` 후 Dense로 가는 편이 낫습니다.
- **2층으로 깊게 쌓기보다** 드롭아웃·유닛 증량이 효율적입니다
  (2층은 하위 LSTM이 `return_sequences=True`로 120 타임스텝 전체를
  상위에 전달해 연산량이 약 2배가 됩니다).
- **다중 출력이 이깁니다**: 온도와 음의 상관인 습도(일주기 패턴이 반대)를
  함께 예측하면 모델이 상반된 일변화를 공동 학습해 정규화 효과를 얻습니다.

## 4. 학습 설정 함정

`rmsprop(1e-3)` + `patience=3` 조합은 val_loss가 9.5 ↔ 18 사이를 요동치게
만들고, 조기 종료가 요동 중간의 나쁜 가중치에 멈춥니다.

```python
# 권장: 낮은 학습률 + 스케줄러 + 넉넉한 patience
optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
callbacks=[keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
           keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)]
```

## 5. 비교 평가 표

여러 모델을 DataFrame 하나로 모아 베이스라인과 나란히 비교합니다.

```python
비교 = {}
for 이름, m in [('MLP', best_mlp), ('Conv1D', best_conv), ('LSTM', best_rnn)]:
    비교[이름] = {split: m.evaluate(dataloader[split], return_dict=True)['mae']
                 for split in dataloader}

비교_df = pd.DataFrame(비교).round(3)
비교_df['baseline'] = pd.Series(baseline_mae).round(3)
display(비교_df)
```

---

## 6. 시계열 트랜스포머 (Time-Series Transformer Encoder)

시계열의 장기 의존성과 다변량 상호작용을 포착하기 위한 트랜스포머 인코더 구현입니다.
Keras 3 (PyTorch 백엔드) 기준 구현 패턴과 실전 주의점입니다.

### 6.1 시계열 트랜스포머 인코더 구현

```python
import keras
from keras import layers

def build_timeseries_transformer(input_shape, num_classes=1, head_size=64,
                                 num_heads=4, ff_dim=128, num_blocks=2, dropout=0.2,
                                 task="regression"):
    """시퀀스 (batch, seq_len, channels) 입력을 위한 트랜스포머 인코더."""
    seq_len, channels = input_shape
    inputs = keras.Input(shape=input_shape)
    
    # 1. 선형 프로젝션
    x = layers.Dense(head_size)(inputs)
    
    # 2. 학습 가능한 위치 임베딩 (Keras 버전에 따라 layers.PositionEmbedding 미지원 시 대응)
    positions = keras.ops.arange(start=0, stop=seq_len, step=1)
    pos_emb = layers.Embedding(input_dim=seq_len, output_dim=head_size, name="pos_emb")(positions)
    x = x + pos_emb
    
    # 3. Transformer 인코더 블록 반복
    for _ in range(num_blocks):
        # Multi-Head Self-Attention + Residual + LayerNorm
        attn_out = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size // num_heads, dropout=dropout
        )(x, x)
        x = layers.Add()([x, attn_out])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-Forward Network + Residual + LayerNorm
        ff = layers.Dense(ff_dim, activation="relu")(x)
        ff = layers.Dropout(dropout)(ff)
        ff = layers.Dense(head_size)(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
    # 4. 출력 헤드
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    
    if task == "regression":
        outputs = layers.Dense(num_classes)(x)
    elif task == "binary":
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else:  # classification
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        
    return keras.Model(inputs, outputs, name="timeseries_transformer")
```

> ⚠️ **위치 임베딩 호환성**: Keras 3 일부 버전에서 `layers.PositionEmbedding` 임포트 오류가 발생할 수 있습니다. 위와 같이 `keras.ops.arange`와 `layers.Embedding`의 조합으로 명시적 학습 위치 임베딩을 구성하면 백엔드(torch/jax/tensorflow) 간 일관성을 유지할 수 있습니다.

---

## 7. 다채널 시계열 분류 실측 벤치마크 (HAR 센서 프로젝트 경험)

3축 가속도계 센서($80$ 시점 $\times 3$축, 6개 활동 레이블 분류) 실습에서 도출된 아키텍처별 실제 성능과 자원 효율성 비교입니다.

| 아키텍처 | Accuracy | F1 (Macro) | 학습 속도/자원 | 실전 특성 및 평가 |
|---|---|---|---|---|
| **1D CNN** (Conv1D 2층 + GAP) | **99.03%** | **0.985** | 가장 빠름 (~30초) | ✅ **최고 성능**. 주기적 파형의 국소 위상/형상 특징 추출에 최적 |
| **CNN + LSTM** | 98.50% | 0.981 | 빠름 | CNN으로 공간/축 특징 압축 후 LSTM으로 시계열 전이 포착 |
| **TCN** (Causal Dilated Conv) | 95.32% | 0.943 | 중간 (병렬화 우수) | 지수적 팽창 컨볼루션으로 넓은 수용영역 확보, 안정적 수렴 |
| **BiLSTM** (2층 양방향) | 95.70% | 0.952 | 느림 | 시퀀스 양방향 문맥 반영하나 연산 오버헤드 큼 |
| **GRU** (2층) | 95.20% | 0.948 | 중간 | LSTM 대비 파라미터가 적고 빠른 수렴 |
| **MOMENT-1-small** (FM 동결+헤드) | 83.11% | 0.688 | 임베딩 추출 느림 | 사전훈련 도메인 불일치 및 인스턴스 정규화에 따른 진폭 손실 |
| **MLP** (평면 요약 통계량) | 62.00% | 0.580 | 매우 빠름 | 시계열 시간 순서 및 파형 역학 완전 상실 |

### 실전 아키텍처 선택 원칙

1. **국소 패턴이 지배적인 고빈도 센서/파형 시계열**:
   - 무조건 트랜스포머나 복잡한 LLM을 쓰기 전에 **1D CNN 또는 TCN**을 우선 베이스라인으로 적용하십시오.
   - 센서 진동, 심전도, 보행 파형 등은 주기적이고 국소적인 형태(Waveform Motif)가 분류의 결정적 단서가 되므로 1D CNN이 트랜스포머보다 파라미터 대비 성능과 수렴 속도가 압도적입니다.
2. **트랜스포머의 적합 영역**:
   - 시간 간격이 불규칙하거나, 수백~수천 스텝에 걸쳐 복잡한 비국소적(Non-local) 원거리 이벤트 간의 상관성을 찾아야 할 때 트랜스포머 인코더가 유효합니다.

---

## 8. 시계열 모델 설명력 (Explainability: Saliency & Gradient × Input)

시계열 예측/분류 모델에서 "어느 시점($t$)"과 "어느 센서 채널($c$)"이 결과에 결정적인 영향을 미쳤는지 설명(XAI)하는 기법입니다.
Keras 3 (PyTorch 백엔드)에서는 Autograd를 직접 활용하여 빠르고 정확하게 그래디언트 살리언시를 산출할 수 있습니다.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_saliency_map(keras_torch_model, x_sample, target_class_idx):
    """
    x_sample: (seq_len, channels) numpy array
    target_class_idx: 예측 대상 클래스 인덱스 (정수)
    반환: (seq_len, channels) 기여도 배열 (Input * Gradient)
    """
    # 1. 텐서 변환 및 그래디언트 추적 활성화
    x_tensor = torch.tensor(x_sample[np.newaxis, ...], dtype=torch.float32, requires_grad=True)
    
    # 2. 순전파
    preds = keras_torch_model(x_tensor)
    target_score = preds[0, target_class_idx]
    
    # 3. 입력에 대한 역전파 (Saliency)
    grads = torch.autograd.grad(target_score, x_tensor)[0]
    
    # 4. Input * Gradient 방식 (기여도 크기 산출)
    attribution = (x_tensor * grads).squeeze(0).detach().cpu().numpy()
    return attribution

def plot_timeseries_saliency(x_sample, attribution, channel_names=None):
    """(seq_len, channels) 형태의 기여도를 히트맵과 파형으로 시각화."""
    seq_len, num_channels = x_sample.shape
    if channel_names is None:
        channel_names = [f"Ch {i}" for i in range(num_channels)]
        
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # 원본 시계열 파형
    for c in range(num_channels):
        axes[0].plot(x_sample[:, c], label=channel_names[c])
    axes[0].set_title("Input Time Series")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    
    # 채널별 시간 기여도 히트맵 (imshow는 (채널, 시간) 형태가 되도록 .T 처리)
    im = axes[1].imshow(attribution.T, aspect="auto", cmap="coolwarm", interpolation="nearest")
    axes[1].set_yticks(range(num_channels))
    axes[1].set_yticklabels(channel_names)
    axes[1].set_title("Feature Contribution (Input × Gradient)")
    axes[1].set_xlabel("Time Step (t)")
    fig.colorbar(im, ax=axes[1], orientation="horizontal", pad=0.2)
    plt.tight_layout()
    plt.show()
```

---

## 9. 주파수-시간 도메인 변환 및 조건부 시계열 생성 (Seismic 프로젝트 경험)

물리 파형(지진파, 음향, 기계 진동) 등 고빈도 비정상(non-stationary) 시계열을 다룰 때는 시간 도메인에서 직접 어텐션을 적용하는 것보다 주파수-시간 변환(STFT/FAS)을 결합하는 것이 수렴과 물리적 정합성에 훨씬 효과적입니다.

1. **시간-주파수 표현 (STFT & FAS)**:
   - 복잡한 위상과 파동 에너지를 단시간 푸리에 변환(STFT)이나 푸리에 진폭 스펙트럼(FAS: Fourier Amplitude Spectrum)으로 변환하여 2D 표현(시간 $\times$ 주파수)으로 축약.
   - 신경망(Dynamic VAE / Causal 인코더)이 진폭 스펙트럼의 포락선(Envelope)과 주파수별 감쇠를 학습하게 하여 긴 시계열($7000+$ 샘플)의 모드 붕괴를 방지.
2. **조건부 메타데이터 주입 (Conditional Broadcasting)**:
   - 시계열의 물리적 발생 조건(지진 규모, 진원 거리, 관측소 좌표 등)을 MLP로 잠재 벡터($d_{cond}$)로 투영한 뒤, 시계열 시퀀스 길이($T$)만큼 브로드캐스팅(`repeat(1, T, 1)`)하여 결합:
   ```python
   # 조건 x: (batch, cond_dim) -> (batch, T, cond_emb_dim)
   cond_emb = mlp_embed(cond_features).unsqueeze(1).repeat(1, seq_len, 1)
   model_inputs = torch.cat([timeseries_features, cond_emb], dim=-1)
   ```
   - 이 방식은 시계열 트랜스포머의 각 타임스텝 토큰에 전역 문맥(Global Conditioning)을 일관되게 주입하는 표준적인 방법입니다.
