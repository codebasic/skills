# 모델 구축

Keras는 세 가지 모델 구축 방식을 제공합니다.

## 1. Sequential API

단순한 순차(레이어를 순서대로 쌓는) 모델에 적합합니다.

```python
import keras
from keras import layers

def build_mlp(input_shape=(28, 28, 1), hidden_size=512, num_classes=10):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(hidden_size, activation='relu'),
        layers.Dense(num_classes),
    ])

model = build_mlp()
model.summary()
```

### 주요 레이어

| 레이어 | 설명 |
|--------|------|
| `layers.Input(shape=...)` | 입력 레이어 (shape에 채널 포함) |
| `layers.Flatten()` | 다차원 입력을 1차원으로 펼침 |
| `layers.Dense(units, activation=...)` | 완전연결(밀집) 레이어 |
| `layers.Conv2D(filters, kernel_size, activation=...)` | 2D 합성곱 레이어 |
| `layers.MaxPooling2D(pool_size)` | 2D 최대 풀링 |
| `layers.Dropout(rate)` | 드롭아웃 (과적합 방지) |
| `layers.BatchNormalization()` | 배치 정규화 |

### 활성화 함수

- `activation='relu'`, `activation='sigmoid'`, `activation='softmax'`, `activation='tanh'`
- 출력층에 활성화 함수를 두지 않으면 `from_logits=True` 사용

## 2. Functional API

분기, 병합, 다중 입력/출력 등 복잡한 구조에 적합합니다.

```python
import keras
from keras import layers

inputs = layers.Input(shape=(28, 28, 1))
x = layers.Flatten()(inputs)
x = layers.Dense(512, activation='relu')(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
```

## 3. Model Subclassing (서브클래싱)

완전한 커스터마이징이 필요할 때 사용합니다. `keras.Model`을 상속합니다.

```python
import keras
from keras import layers

class MLP(keras.Model):
    def __init__(self, hidden_size=512, num_classes=10):
        super().__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(hidden_size, activation='relu')
        self.dense2 = layers.Dense(num_classes)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x)

model = MLP()
model.build((None, 28, 28, 1))
model.summary()
```

## 모델 정보 확인

```python
model.summary()   # 레이어 구조, 파라미터 수 출력
```

### 파라미터 수 계산

```python
import math

def count_params(model):
    return sum(math.prod(p.shape) for p in model.trainable_variables)

print(f'Parameter count: {count_params(model):,}')
```

## 입력/출력 확인

PyTorch 백엔드에서는 모델에 `torch.Tensor`를 직접 입력할 수 있습니다.

```python
import torch

# Keras 3 기본 채널-마지막(HWC) 포맷: (batch, height, width, channels)
dummy = torch.randn(1, 28, 28, 1)
output = model(dummy)
print(f'Input shape: {tuple(dummy.shape)}')
print(f'Output shape: {tuple(output.shape)}')
print(f'Output type: {type(output)}')  # torch.Tensor
```

## MNIST 예제 (Sequential)

```python
import keras
from keras import layers

def build_keras_mlp(input_shape=(28, 28, 1), hidden_size=512, num_classes=10):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(hidden_size, activation='relu'),
        layers.Dense(num_classes),
    ])
```

> 💡 **팁**: 모델을 반복적으로 활용하고 전역 네임스페이스 오염을 피하려면 모델 구축을 함수로 정의하세요.
