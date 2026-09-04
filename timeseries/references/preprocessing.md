# 전처리: 적재 → 정제 → 리샘플링 → 분할 → 표준화

## 1. 적재와 정렬

datetime 열을 인덱스로 읽고 **반드시 정렬 + 단조성 검증**부터 합니다.
시계열 분할·윈도잉은 전부 "인덱스 순서 = 시간 순서" 가정 위에 서 있습니다.

```python
from pathlib import Path

파일경로 = Path('data/jena_climate_2009_2016.csv')
weather = pd.read_csv(
    파일경로,
    parse_dates=['Date Time'],
    date_format=r'%d.%m.%Y %H:%M:%S',
    index_col='Date Time')

weather.sort_index(inplace=True)
assert weather.index.is_monotonic_increasing  # 단조 증가 확인
```

## 2. 센서 오류값 처리

실측 데이터에는 물리적으로 불가능한 값이 섞여 있습니다. Jena 풍속 열에는
`-9999.0` 센서 오류가 존재하며, **테스트 구간에 포함되어 있으면 모델 예측이 폭주**
합니다(실제로 표준화 후 극단 입력으로 테스트 loss가 수천 배 폭등한 사례 있음).

```python
# 풍속 센서 오류값(-9999.0)을 0으로 교체
for col in ['wv (m/s)', 'max. wv (m/s)']:
    weather.loc[weather[col] == -9999.0, col] = 0.0
```

> 💡 적재 직후 `min/max`를 확인해 물리적으로 불가능한 값(-9999, -1e30 등)을 찾으세요.
> 오류값이 훈련 구간에만 있으면 학습이 망가지고, 평가 구간에만 있으면 평가가 망가집니다.

## 3. 리샘플링

10분 간격 → 1시간 단위 다운샘플링. 결측 구간(리샘플 후 빈 구간)은 제거합니다.

```python
weather_1H = weather.resample('1h').mean().dropna()
```

## 4. 분할 — 시간 순서 유지 (절대 섞지 않음)

`train_test_split(shuffle=False)`로 시간 순서를 유지한 채 앞쪽 50% 훈련,
나머지를 반으로 나눠 검증/테스트로 사용합니다.

```python
dataset = {}
dataset['train'], dataset['test'] = sklearn.model_selection.train_test_split(
    weather_1H, test_size=0.5, shuffle=False)
dataset['val'], dataset['test'] = sklearn.model_selection.train_test_split(
    dataset['test'], test_size=0.5, shuffle=False)
```

> ⚠️ 시계열에서 `shuffle=True`는 미래 정보 누수(look-ahead)를 유발합니다.

## 5. 표준화

**훈련 데이터로만** `StandardScaler`를 학습(fit)하고 세 분할에 적용합니다.

```python
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(dataset['train'])

scaled_dataset = {split: scaler.transform(dataset[split]) for split in dataset}

# 훈련 데이터 기준 평균 0, 표준편차 1 확인
assert (np.isclose(scaled_dataset['train'].mean(), 0.0) and
        np.isclose(scaled_dataset['train'].std(), 1.0))
```

### 전역 인덱싱이 필요할 때

전체 시간축을 기준으로 윈도우를 슬라이싱해야 하는 경우(파운데이션 모델 평가 등),
분할별 배열(`scaled_dataset[split]`)은 각자 0부터 시작하는 **로컬 인덱스**라
전역 시점 인덱스로 슬라이싱하면 길이가 맞지 않아 오류가 납니다. 이때는
전체 구간을 한 번에 표준화한 배열을 별도로 만들어 사용합니다.

```python
scaled_전체 = scaler.transform(weather_1H)  # 전역 인덱싱용
x = scaled_전체[s - ctx:s]                  # 전역 시점 s 기준 윈도우
```

### 원래 단위 복원

예측 평가는 해석이 쉬운 원래 단위(°C)로 합니다. 표준화된 값 `z`에서
원래 값으로의 복원: `z * scale + mean`.

```python
온도열위치 = weather_1H.columns.get_loc('T (degC)')
복원값 = 표준값 * scaler.scale_[온도열위치] + scaler.mean_[온도열위치]
# 또는 행 단위: scaler.inverse_transform(행)
```

## 체크리스트

- [ ] 인덱스 정렬 + 단조성 assert
- [ ] 물리적으로 불가능한 센서 오류값 탐지·교체 (min/max 확인)
- [ ] 리샘플링 + 결측 제거
- [ ] `shuffle=False` 분할
- [ ] 훈련 데이터로만 scaler fit
- [ ] 평가 지표는 원래 단위(°C)로 역변환 후 계산
