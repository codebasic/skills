---
name: keras
description: 'Keras 3를 PyTorch 백엔드로 사용하는 방법을 안내하는 스킬. Keras 모델 구축(Sequential/Functional/서브클래싱), 컴파일, 훈련, 평가, PyTorch DataLoader 활용, MNIST 이미지 분류 실습을 다룹니다. Use when: Keras 코드 작성, keras.Sequential, keras.layers, model.compile, model.fit, KERAS_BACKEND=torch 설정, 파이토치 백엔드 케라스 사용, DataLoader, torch.utils.data, MNIST 분류 모델 구축.'
argument-hint: 'Keras + PyTorch 백엔드 사용법'
user-invocable: true
---

# Keras 3 (PyTorch 백엔드)

Keras 3는 다중 백엔드(멀티-백엔드)를 지원하는 딥러닝 API입니다. JAX, TensorFlow, PyTorch 중 하나를 백엔드로 선택할 수 있으며, **PyTorch 백엔드**를 사용하면 텐서플로우 없이도 Keras의 고수준 API를 활용할 수 있습니다.

## When to Use

- Keras로 모델을 구축하거나 훈련해야 할 때
- PyTorch 백엔드(`KERAS_BACKEND=torch`)로 Keras 3를 사용할 때
- MNIST 등 이미지 분류 모델을 Keras로 작성할 때
- `keras.Sequential`, `keras.layers`, `model.compile`, `model.fit` 사용법이 필요할 때

## 핵심 개념

Keras 3에서 PyTorch 백엔드를 사용하면:

- `import keras` 만으로 사용 가능 (텐서플로우 불필요)
- 텐서는 PyTorch 텐서(`torch.Tensor`)로 표현됨
- `keras.backend.backend()`로 현재 백엔드 확인 가능
- 모델에 PyTorch 텐서를 직접 입력할 수 있음

## 코드 작성 스타일 (노트북 컨벤션)

이 워크스페이스의 노트북 코드 스타일입니다. Keras 관련 코드를 작성할 때 이 컨벤션을 따릅니다.

### 1. 반복 작업은 유틸리티 함수로 추출

학습·평가·데이터로더 구성처럼 여러 모델/데이터셋에서 반복되는 작업은 유틸리티 함수로 추출합니다. 유틸리티 함수는 **import 셀 바로 다음 셀**에 정의합니다.

```python
def create_dataloaders(datasets, **kwargs):
    """데이터로더를 구성합니다.

    Args:
        datasets: 데이터셋 딕셔너리
        **kwargs: DataLoader에 전달할 키워드 인자 (batch_size, num_workers 등)

    Returns:
        데이터로더 딕셔너리
    """
    dataloaders = {}
    for split in datasets:
        dataloaders[split] = DataLoader(
            datasets[split],
            shuffle=split == 'train',   # shuffle은 train 여부로 자동 결정
            **kwargs,
        )
    return dataloaders


def evaluate_model(model, dataloader):
    """모델을 평가합니다."""
    scores = {}
    for split in dataloader:
        scores[split] = model.evaluate(dataloader[split], return_dict=True)
    return scores
```

### 2. 인자는 키워드 인자로, 하드코딩 금지

함수에 전달하는 설정(batch_size, num_workers, epochs 등)은 하드코딩하지 말고 키워드 인자로 받습니다. 함수 내부에서 자동으로 결정되는 값(shuffle 등)만 내부 로직으로 처리합니다.

```python
# ✅ 권장
dataloader = create_dataloaders(fashion_mnist, batch_size=100, num_workers=2)
history = model.fit(dataloader['train'], epochs=10)

# ❌ 지양 — 설정이 함수 시그니처에 하드코딩됨
def create_dataloaders(datasets, batch_size=100, num_workers=2, shuffle_train=True): ...
```

### 3. 모델 구축도 함수로 추출

모델 구축은 `build_model()` 형태의 함수로 추출합니다. 하이퍼파라미터는 인자로 받고, 구조가 달라지는 조건(예: 공간 차원에 따른 풀링 적용)은 내부 로직으로 처리합니다.

```python
def build_model():
    inputs = keras.Input(shape=(28, 28, 1))
    x = inputs
    for filters in [32, 64, 128]:
        x = keras.layers.Conv2D(
            filters=filters, kernel_size=3, activation='relu')(x)
        # 입력의 공간적 차원이 너무 작지 않은 경우에만 풀링 적용
        if x.shape[1] > 3:
            x = keras.layers.MaxPooling2D(pool_size=2)(x)

    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs)
```

### 4. 성능 비교는 DataFrame으로 표시

여러 모델/전처리의 결과를 비교할 때는 split별 `evaluate(..., return_dict=True)` 결과를 `pd.DataFrame`으로 정리해 `display()`로 표시합니다.

```python
scores = evaluate_model(model, dataloader)
display(pd.DataFrame(scores).round(3))
```

### 5. 커스텀 Dataset은 파일 목록을 받도록 일반화

커스텀 `Dataset`은 특정 폴더에 종속되지 않도록 **파일 경로 목록**을 받습니다. 클래스 라벨은 파일의 부모 폴더명에서 자동으로 추출합니다. (완전 구현은 [scripts/mnist_classification.py](./scripts/mnist_classification.py)와 워크스페이스 `image_datasets.py` 참조)

```python
class ImageDataset(Dataset):
    def __init__(self, files, transform=None):
        """이미지 데이터셋을 초기화합니다.

        Args:
            files: 이미지 파일 경로 목록 (클래스는 부모 폴더명으로 결정)
            transform (callable, optional): 이미지에 적용할 전처리 함수
        """
        self.files = list(files)
        self.classes = sorted({Path(f).parent.name for f in self.files})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = [(Path(f), self.class_to_idx[Path(f).parent.name]) for f in self.files]
        self.transform = transform
```

### 6. 기타 컨벤션

- 한글 변수명(예: `전처리`, `개냥이폴더`)과 한글 주석/docstring을 사용할 수 있습니다. 주석은 코드의 의도를 설명합니다.
- 전처리 파이프라인은 `torchvision.transforms.Compose`로 구성하며, 새 전처리는 기존 파이프라인을 **재활용해 추가**합니다 (처음부터 다시 정의하지 않음).
- 시드가 필요한 무작위 연산(픽셀 셔플 등)은 함수 내부에서 시드를 고정해 재현성을 보장합니다.

## Procedure

### 0. conda 환경 활성화

이 워크스페이스에서는 시스템의 conda `pytorch` 환경을 사용합니다. 해당 환경에는 Keras 3.10.0과 PyTorch 2.8.0(CUDA 지원)이 이미 설치되어 있습니다.

```bash
conda activate pytorch
```

환경 확인:
```bash
python -c "import keras; print(keras.__version__, keras.backend.backend())"
# 예: 3.10.0 torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 예: 2.8.0+cu128 True
```

### 1. 백엔드 설정

백엔드는 **Keras를 import 하기 전에** 설정해야 합니다. 세 가지 방법이 있습니다:

**방법 A - 환경 변수 (권장):**
```bash
export KERAS_BACKEND="torch"
```

**방법 B - 코드 내 설정 (노트북/스크립트):**
```python
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
```

**방법 C - 설정 파일** (`~/.keras/keras.json`):
```json
{
    "backend": "torch"
}
```

> ⚠️ 백엔드는 Keras를 import 한 뒤에는 변경할 수 없습니다. 반드시 import 전에 설정하세요.

### 2. 설치

```bash
pip install --upgrade keras
pip install torch torchvision   # PyTorch 백엔드
```

버전 확인:
```python
import keras
print(keras.__version__)          # Keras 버전
print(keras.backend.backend())    # 현재 백엔드 (예: 'torch')
```

### 3. 모델 구축

Keras는 세 가지 모델 구축 방식을 제공합니다. 자세한 내용은 [model-building.md](./references/model-building.md) 참조.

**Sequential API (간단한 순차 모델):**
```python
import keras
from keras import layers

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(10),
])
model.summary()
```

### 4. 모델 컴파일

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)
```

### 5. 모델 훈련

```python
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=5,
    validation_data=(x_test, y_test),
    verbose=1,
)
```

### 6. PyTorch DataLoader 활용

Keras 3는 PyTorch 백엔드에서 `torch.utils.data.DataLoader`를 `model.fit()`에 **직접 전달**할 수 있습니다. 대용량 데이터셋, 커스텀 데이터셋, 증강 파이프라인에 유용합니다. 자세한 내용은 [training.md](./references/training.md) 참조.

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# DataLoader를 model.fit()에 직접 전달 (batch_size 지정하지 않음)
history = model.fit(
    train_loader,
    epochs=5,
    validation_data=test_loader,
    verbose=1,
)
```

> ⚠️ DataLoader를 사용할 때는 `model.fit()`에 `batch_size`를 **지정하지 마세요**. 배치 크기는 DataLoader에서 결정됩니다.

### 7. 평가 및 예측

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
predictions = model.predict(x_test)
```

## 참고 자료

- [설치 및 백엔드 설정](./references/installation.md)
- [모델 구축 (Sequential/Functional/서브클래싱)](./references/model-building.md)
- [컴파일, 훈련, 평가, PyTorch DataLoader](./references/training.md)
- [MNIST 분류 예제 스크립트](./scripts/mnist_classification.py)

## 주의사항

- 백엔드는 import 전에 설정해야 하며, 이후 변경 불가
- PyTorch 백엔드에서 모델 출력은 `torch.Tensor` 타입
- `from_logits=True`는 출력층에 활성화 함수(softmax)가 없을 때 사용
- MNIST 데이터는 `keras.datasets.mnist.load_data()`로 로드 가능
- torchvision `datasets.MNIST`는 `(1, 28, 28)` 채널-우선(CHW) 형태, `keras.datasets.mnist`는 `(28, 28, 1)` 채널-마지막(HWC) 형태 — 모델 `input_shape`을 이에 맞춰야 함
