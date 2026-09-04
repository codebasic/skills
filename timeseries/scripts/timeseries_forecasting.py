"""
시계열 예측 전체 파이프라인 참조 구현 (Jena 기후 데이터)

노트북 컨벤션에 따라 작성된 예제:
    - 전처리(정렬, 센서 오류 교체, 리샘플링, 분할, 표준화)
    - 슬라이딩 윈도우 시퀀스 데이터셋 (제너레이터 1회성 주의)
    - persistence 베이스라인
    - LSTM 다중출력 모델 (베이스라인을 상회한 최고 성능 구성)
    - 체크포인트 저장 후 로드해 평가

사용법 (conda pytorch 환경):
    conda activate pytorch
    export KERAS_BACKEND="torch"
    python timeseries_forecasting.py             # 전체 학습 (약 10분)
    SMOKE_TEST=1 python timeseries_forecasting.py  # 1 에포크 smoke test (약 1분)
"""
import os

os.environ["KERAS_BACKEND"] = "torch"  # Keras import 전에 설정

from pathlib import Path

import keras
import numpy as np
import pandas as pd
import sklearn
import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# 유틸리티 (import 직후 정의)
# ---------------------------------------------------------------------------

def timeseries_dataset_from_array(dataset, sequence_length, target=None, delay=1):
    """슬라이딩 윈도우 (입력 시퀀스, 지연된 타깃) 쌍을 생성하는 제너레이터.

    Args:
        dataset: (시간, 특성) 2차원 배열
        sequence_length: 입력 시퀀스 길이
        target: 데이터셋과 같은 시점 기준 타깃 배열 (None이면 dataset 자체)
        delay: 윈도우 끝에서 delay만큼 뒤의 값을 타깃으로 사용

    Yields:
        (입력 시퀀스, 타깃) 튜플
    """
    if target is None:
        target = dataset
    for i in range(len(dataset) - sequence_length - delay + 1):
        yield dataset[i:i + sequence_length], target[i + sequence_length - 1 + delay]


def build_dataloaders(seq_datasets, **kwargs):
    """시퀀스 제너레이터 딕셔너리를 TensorDataset 기반 DataLoader로 변환합니다."""
    dataloaders = {}
    for split, seq_ds in seq_datasets.items():
        samples = list(seq_ds)  # 제너레이터 소진
        x = torch.tensor(np.array([s for s, _ in samples]), dtype=torch.float32)
        y = torch.tensor(np.array([t for _, t in samples]), dtype=torch.float32)
        dataloaders[split] = DataLoader(
            TensorDataset(x, y), shuffle=split == 'train', **kwargs)
    return dataloaders


# ---------------------------------------------------------------------------
# 1. 전처리
# ---------------------------------------------------------------------------

파일경로 = Path(__file__).resolve().parents[3] / 'data' / 'jena_climate_2009_2016.csv'
weather = pd.read_csv(
    파일경로, parse_dates=['Date Time'],
    date_format=r'%d.%m.%Y %H:%M:%S', index_col='Date Time')

# 정렬 + 단조성 검증
weather.sort_index(inplace=True)
assert weather.index.is_monotonic_increasing

# 센서 오류값(-9999.0) 교체 — 평가/훈련 구간 어디든 폭발적 오류의 원인
for col in ['wv (m/s)', 'max. wv (m/s)']:
    weather.loc[weather[col] == -9999.0, col] = 0.0

# 1시간 리샘플링 + 결측 제거
weather_1H = weather.resample('1h').mean().dropna()

# 시간 순서 유지 분할 (절대 섞지 않음)
dataset = {}
dataset['train'], dataset['test'] = sklearn.model_selection.train_test_split(
    weather_1H, test_size=0.5, shuffle=False)
dataset['val'], dataset['test'] = sklearn.model_selection.train_test_split(
    dataset['test'], test_size=0.5, shuffle=False)

# 훈련 데이터로만 표준화
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(dataset['train'])
scaled_dataset = {split: scaler.transform(dataset[split]) for split in dataset}

온도열위치 = weather_1H.columns.get_loc('T (degC)')
습도열위치 = weather_1H.columns.get_loc('rh (%)')
시퀀스길이, 특성수 = 24 * 5, len(weather_1H.columns)

# ---------------------------------------------------------------------------
# 2. 시퀀스 데이터셋 (다중 출력: 온도 + 습도) — 항상 새로 생성할 것 (제너레이터 1회성)
# ---------------------------------------------------------------------------

타깃위치 = [온도열위치, 습도열위치]
seq_dataset = {}
for split in scaled_dataset:
    seq_dataset[split] = timeseries_dataset_from_array(
        scaled_dataset[split], 시퀀스길이,
        target=scaled_dataset[split][:, 타깃위치],  # 표준화된 (N, 2) 타깃
        delay=24)

dataloader = build_dataloaders(seq_dataset, batch_size=512)

# ---------------------------------------------------------------------------
# 3. 베이스라인 (persistence) — 모델링 전 반드시 계산
# ---------------------------------------------------------------------------

baseline_mae = {}
for split in scaled_dataset:
    starts = scaled_dataset[split][시퀀스길이 - 1:-24]
    targets = scaled_dataset[split][시퀀스길이 - 1 + 24:, [온도열위치, 습도열위치]]
    mae = np.mean(np.abs(starts[:, 타깃위치] - targets), axis=0)
    baseline_mae[split] = dict(zip(['온도 (°C)', '상대습도 (%)'], mae))
print('baseline MAE:', pd.DataFrame(baseline_mae).round(3))

# ---------------------------------------------------------------------------
# 4. 모델: LSTM 다중출력 (온도 + 상대습도)
# ---------------------------------------------------------------------------

def build_multi_rnn():
    """온도와 상대습도를 동시에 예측하는 다중 출력 LSTM."""
    inputs = keras.Input(shape=(시퀀스길이, 특성수))
    x = keras.layers.LSTM(16)(inputs)
    outputs = keras.layers.Dense(2)(x)  # [온도, 상대습도] 표준화된 값
    return keras.Model(inputs, outputs, name='lstm_multi')


keras.backend.clear_session()
model = build_multi_rnn()
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
              loss=keras.losses.MeanSquaredError(), metrics=['mae'])

체크포인트폴더 = Path(__file__).resolve().parents[3] / 'checkpoints'
체크포인트폴더.mkdir(exist_ok=True)

# smoke test: SMOKE_TEST=1 환경변수로 1 에포크만 학습해 파이프라인 동작을 빠르게 확인
SMOKE_TEST = os.environ.get('SMOKE_TEST') == '1'
에포크 = 1 if SMOKE_TEST else 30
조기종료_인내 = 1 if SMOKE_TEST else 7
if SMOKE_TEST:
    print('[SMOKE TEST] 1 에포크만 학습합니다.')

model.fit(
    dataloader['train'],
    validation_data=dataloader['val'],
    epochs=에포크,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            filepath=체크포인트폴더 / 'lstm_multi.keras',
            monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(patience=조기종료_인내, restore_best_weights=True),
    ],
)

# ---------------------------------------------------------------------------
# 5. 평가 — 저장된 체크포인트를 로드해 원래 단위(°C, %) MAE로 계산
# ---------------------------------------------------------------------------

best_model = keras.models.load_model(체크포인트폴더 / 'lstm_multi.keras')


def 표준값_복원(표준값, 열위치):
    """표준화된 값을 원래 단위로 복원."""
    return 표준값 * scaler.scale_[열위치] + scaler.mean_[열위치]


평가rows = []
for split in dataloader:
    예측, 목표, 마지막행 = [], [], []
    for x_batch, y_batch in dataloader[split]:
        예측.append(best_model.predict(x_batch, verbose=0))
        목표.append(y_batch.numpy())
        마지막행.append(x_batch[:, -1, :].numpy())
    예측, 목표, 마지막행 = map(np.concatenate, (예측, 목표, 마지막행))

    for j, (이름, 위치) in enumerate(
            [('온도 (°C)', 온도열위치), ('상대습도 (%)', 습도열위치)]):
        목표값 = 표준값_복원(목표[:, j], 위치)
        평가rows.append({
            'split': split, '타깃': 이름,
            'LSTM 다중출력': np.mean(np.abs(표준값_복원(예측[:, j], 위치) - 목표값)),
            'baseline': np.mean(np.abs(표준값_복원(마지막행[:, 위치], 위치) - 목표값)),
        })

평가_df = pd.DataFrame(평가rows).set_index(['타깃', 'split']).round(3)
print(평가_df)
print('\n기대 결과: 온도 val ~2.35, test ~2.50 °C — baseline(val 2.43, test 2.61) 상회')
