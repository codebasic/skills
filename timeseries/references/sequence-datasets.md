# 시퀀스 데이터셋: 슬라이딩 윈도우 생성

## 개념

과거 `sequence_length` 시점을 입력으로, 그 끝에서 `delay`만큼 뒤의 값을 타깃으로
하는 (입력 시퀀스, 타깃) 쌍을 슬라이딩 윈도우 방식으로 생성합니다.

- 입력: 과거 **5일(24 × 5시간)** 분량의 다변량 시퀀스 `(120, 14)`
- 타깃: 윈도우 마지막 시점에서 **24시간 뒤**의 값

## 구현 (timeseries_datasets 모듈)

```python
# timeseries_datasets.py — 제너레이터 기반 유틸리티
def create_sequence_dataset(dataset, sequence_length, offset=1):
    """슬라이딩 윈도우 방식으로 시퀀스를 생성하는 제너레이터."""
    for i in range(0, len(dataset) - sequence_length, offset):
        yield dataset[i:i + sequence_length]

def timeseries_dataset_from_array(dataset, sequence_length, target=None, delay=1):
    """(입력 시퀀스, 지연된 타깃) 쌍을 생성하는 제너레이터."""
    if target is None:
        target = dataset
    for i, seq in enumerate(
            create_sequence_dataset(dataset[:len(dataset) - delay + 1], sequence_length)):
        y = target[i + sequence_length - 1 + delay]  # 윈도우 끝 + delay 시점
        yield seq, y
```

## 사용 예

```python
seq_dataset = {}
for split in scaled_dataset:
    seq_dataset[split] = timeseries_dataset_from_array(
        scaled_dataset[split], sequence_length,           # (120, 14) 입력
        target=scaler.inverse_transform(scaled_dataset[split])[:, 온도열위치],  # °C 타깃
        delay=24)                                          # 24시간 뒤

for split in seq_dataset:
    for x, y in seq_dataset[split]:
        print(f"{split}: 입력 {x.shape} -> 목표 {y:.3f} °C")
        break
```

## ⚠️ 제너레이터는 1회성

제너레이터를 한 번 순회하면 **소진**됩니다. 베이스라인 계산에서
`samples = list(seq_dataset[split])`로 소진한 뒤 같은 제너레이터를
모델 학습에 재사용하면 빈 데이터셋이 됩니다.

```python
# ❌ 잘못된 흐름
baseline 계산에 seq_dataset 사용 (소진됨)
model.fit(seq_dataset)  # → 빈 데이터셋

# ✅ 올바른 흐름: 필요할 때마다 다시 생성
seq_dataset = {split: timeseries_dataset_from_array(...) for split in ...}
```

## 텐서 변환 + DataLoader

Keras `model.fit`은 제너레이터를 직접 받을 수 있지만, 배치 통계 확인이나
GPU 가동률 개선(배치 확대)을 위해 텐서로 변환해 `DataLoader`로 감싸는 것이
편리합니다.

```python
from torch.utils.data import DataLoader, TensorDataset

def build_dataloaders(seq_dataset, batch_size=512):
    """시퀀스 제너레이터를 텐서 데이터셋으로 변환한 뒤 DataLoader를 구성합니다."""
    dataloaders = {}
    for split in seq_dataset:
        samples = list(seq_dataset[split])
        x = torch.tensor(np.array([s for s, _ in samples]), dtype=torch.float32)
        y = torch.tensor(np.array([t for _, t in samples]), dtype=torch.float32)
        dataloaders[split] = DataLoader(
            TensorDataset(x, y), batch_size=batch_size,
            shuffle=split == 'train')
    return dataloaders
```

> 💡 배치 크기를 128 → 512로 확대하면 GPU 가동률이 눈에 띄게 올라갑니다
> (측정: 평균 18%대, 최대 58%). 다만 모델이 작으면 GPU보다 CPU 오버헤드가
> 병목이 되는 한계는 있습니다.

## 다중 타깃

온도와 습도처럼 상관된 변수를 동시에 예측하면(다중 출력) 단일 출력보다
성능이 나아질 수 있습니다(실측: 단일 온도 val 2.455 → 온도+습도 val 2.351).
공변량 정보를 타깃으로도 활용하고, 공동 학습의 정규화 효과가 작용합니다.

```python
다중타깃위치 = [온도열위치, 습도열위치]
multi_seq_dataset = {
    split: timeseries_dataset_from_array(
        scaled_dataset[split], sequence_length,
        target=scaled_dataset[split][:, 다중타깃위치],  # (N, 2) 표준화된 타깃
        delay=24)
    for split in scaled_dataset}
# 출력층: keras.layers.Dense(2) — 표준화된 값으로 회귀 후 평가 시 역변환
```
