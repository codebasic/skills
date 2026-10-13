# 컴파일, 훈련, 평가

## 데이터 준비 (MNIST)

```python
import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 공식 훈련 집합에서 검증 집합을 따로 분리합니다.
rng = np.random.default_rng(42)
indices = rng.permutation(len(x_train))
train_indices, val_indices = indices[6000:], indices[:6000]
x_val, y_val = x_train[val_indices], y_train[val_indices]
x_train, y_train = x_train[train_indices], y_train[train_indices]

# 정규화 (0~1)
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 채널 차원 추가 (28, 28) -> (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_val = np.expand_dims(x_val, -1)
x_test = np.expand_dims(x_test, -1)

print(f'Train shape: {x_train.shape}, Test shape: {x_test.shape}')
```

검증 집합은 epoch·하이퍼파라미터·체크포인트 선택에 사용하고 공식 시험 집합은 선택이 끝난 뒤 평가합니다. 고정 상수 255로 나누는 변환은 시험 표본의 통계를 학습하지 않습니다. 시험 데이터를 단순히 `validation_data`로 전달했다고 gradient 학습에 쓰이는 것은 아니지만, 그 결과로 조기 종료·체크포인트를 선택하거나 설정을 바꾸면 독립적인 최종 평가가 사라집니다.

## 컴파일

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)
```

### 옵티마이저

- `keras.optimizers.Adam(learning_rate=...)`
- `keras.optimizers.SGD(learning_rate=...)`
- `keras.optimizers.RMSprop(learning_rate=...)`

### 손실 함수

| 손실 | 사용처 |
|------|--------|
| `SparseCategoricalCrossentropy(from_logits=True)` | 정수 라벨 다중 분류 (출력층에 softmax 없을 때) |
| `SparseCategoricalCrossentropy()` | 정수 라벨 다중 분류 (출력층에 softmax 있을 때) |
| `CategoricalCrossentropy()` | 원-핫 라벨 다중 분류 |
| `BinaryCrossentropy()` | 이진 분류 |
| `MeanSquaredError()` | 회귀 |

> 💡 `from_logits=True`는 출력층에 활성화 함수(softmax)가 없을 때 사용합니다.

## 훈련

```python
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=5,
    validation_data=(x_val, y_val),
    verbose=1,
)
```

### 반환값 `history`

`history.history`는 에포크별 훈련 지표를 담은 딕셔너리입니다.

```python
print(history.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
```

### 콜백 (선택)

```python
callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True),
]

history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
)
```

## PyTorch DataLoader 활용

Keras 3는 PyTorch 백엔드에서 `torch.utils.data.DataLoader`를 **`model.fit()`에 직접 전달**할 수 있습니다. 이 방식은 대용량 데이터셋, 커스텀 데이터셋, 증강(augmentation) 파이프라인을 사용할 때 유용합니다.

### 1. torchvision Dataset + DataLoader

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_dataset, val_dataset = random_split(
    train_dataset, [54000, 6000], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

### 2. 커스텀 Dataset 정의

```python
import torch
from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = MNISTDataset(x_train, y_train)
val_dataset = MNISTDataset(x_val, y_val)
test_dataset = MNISTDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

### 3. DataLoader로 훈련

`model.fit()`에 DataLoader를 직접 전달합니다. 이 경우 `batch_size`를 지정하지 않습니다 (DataLoader가 배치를 처리).

```python
history = model.fit(
    train_loader,
    epochs=5,
    validation_data=val_loader,
    verbose=1,
)
```

> ⚠️ DataLoader를 사용할 때는 `model.fit()`에 `batch_size`를 **지정하지 마세요**. 배치 크기는 DataLoader에서 이미 결정됩니다.

### 4. DataLoader로 평가/예측

```python
test_loss, test_acc = model.evaluate(test_loader)
print(f'Test accuracy: {test_acc:.4f}')

predictions = model.predict(test_loader)
```

### 전체 예제 (DataLoader + MNIST)

```python
import os
os.environ['KERAS_BACKEND'] = 'torch'
import torch
import keras
from keras import layers
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

def build_keras_mlp(input_shape=(1, 28, 28), hidden_size=512, num_classes=10):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(hidden_size, activation='relu'),
        layers.Dense(num_classes),
    ])

def train_keras_dataloader(epochs=5, batch_size=64, lr=0.001):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_dataset, val_dataset = random_split(
        train_dataset, [54000, 6000], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = build_keras_mlp(input_shape=(1, 28, 28))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    model.summary()

    history = model.fit(
        train_loader,
        epochs=epochs,
        validation_data=val_loader,
        verbose=1,
    )
    print('Final test:', model.evaluate(test_loader, return_dict=True))
    return model, history

model, history = train_keras_dataloader(epochs=5)
```

> 💡 **팁**: torchvision의 `datasets.MNIST`는 텐서를 `(1, 28, 28)` 채널-우선(CHW) 형태로 반환하므로, 모델의 `input_shape`도 `(1, 28, 28)`로 맞춰야 합니다. numpy 배열(`keras.datasets.mnist`)을 사용할 때는 `(28, 28, 1)` 채널-마지막(HWC) 형태입니다.

## 평가

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')
```

## 예측

```python
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
```

## 전체 예제 (MNIST MLP)

```python
import os
os.environ['KERAS_BACKEND'] = 'torch'
import numpy as np
import keras
from keras import layers
from keras.datasets import mnist

def build_keras_mlp(input_shape=(28, 28, 1), hidden_size=512, num_classes=10):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(hidden_size, activation='relu'),
        layers.Dense(num_classes),
    ])

def train_keras(epochs=5, batch_size=64, lr=0.001):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    indices = np.random.default_rng(42).permutation(len(x_train))
    train_indices, val_indices = indices[6000:], indices[:6000]
    x_val, y_val = x_train[val_indices], y_train[val_indices]
    x_train, y_train = x_train[train_indices], y_train[train_indices]

    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_val = np.expand_dims(x_val, -1)
    x_test = np.expand_dims(x_test, -1)

    model = build_keras_mlp()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    model.summary()

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=1,
    )
    print('Final test:', model.evaluate(x_test, y_test, return_dict=True))
    return model, history

model, history = train_keras(epochs=5)
```

> 💡 **팁**: 훈련 로직도 함수로 캡슐화하면 재사용이 쉽고 전역 네임스페이스 오염을 피할 수 있습니다.
