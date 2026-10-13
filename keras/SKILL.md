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

### 0. Pre-flight Check (환경 진단)

작업 전 Keras 3 버전과 PyTorch 백엔드 연결 상태를 5초 내에 확인합니다:

```bash
python -c "import os; os.environ['KERAS_BACKEND']='torch'; import keras, torch; \
print(f'Keras: {keras.__version__} (Backend: {keras.backend.backend()})'); \
print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()}, MPS: {getattr(torch.backends, \"mps\", None) and torch.backends.mps.is_available()})')"
```

### 1. 백엔드 설정

백엔드는 **Keras를 import 하기 전에** 반드시 설정해야 합니다:

**방법 A - 환경 변수 (터미널/권장):**
```bash
export KERAS_BACKEND="torch"
```

**방법 B - 코드 내 설정 (노트북/스크립트 상단):**
```python
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
```

> ⚠️ Keras를 임포트한 이후에는 백엔드를 변경할 수 없습니다.

### 2. 핵심 워크플로우 요약

상세 가이드와 아키텍처별 코드는 `references/` 문서를 참조하세요:

1. **모델 구축**: [model-building.md](./references/model-building.md)
   - Sequential, Functional, 서브클래싱 3가지 방식 지원
   - 모델 구축은 항상 `build_model()` 함수로 캡슐화
2. **컴파일 & 손실 함수**: [training.md](./references/training.md)
   - 출력층에 Softmax가 없으면 `from_logits=True` 지정
3. **PyTorch DataLoader 연동**: [training.md](./references/training.md)
   - `model.fit(train_loader, epochs=...)`로 직접 전달
   - ⚠️ DataLoader 사용 시 `model.fit()`에 `batch_size`를 **절대 지정하지 않음**
4. **평가 및 성능 비교**:
   - `model.evaluate(dataloader, return_dict=True)` 결과를 `pd.DataFrame`으로 취합
   - 모델·epoch·하이퍼파라미터·임계값 선택에는 검증 집합을 사용하고, 시험 집합은 선택을 마친 뒤 최종 평가에 사용합니다. 시험 성능표를 본 뒤 후보를 선택하거나 설정을 바꾸면 그 표는 독립적인 최종 성능 추정이 아닙니다.
   - `scripts/mnist_classification.py`는 미리 정한 MLP와 CNN을 고정 epoch로 훈련하여 시험 결과를 나란히 표시하는 예제입니다. 표를 모델 선택에 활용하려면 별도 검증 집합에서 비교·선택하고 고정한 모델을 시험 집합에서 평가하도록 확장합니다.

완전한 실행 예제는 [scripts/mnist_classification.py](./scripts/mnist_classification.py)를 참조하세요.

## 참고 자료

- [설치 및 백엔드 설정](./references/installation.md)
- [모델 구축 (Sequential/Functional/서브클래싱)](./references/model-building.md)
- [컴파일, 훈련, 평가, PyTorch DataLoader](./references/training.md)
- [MNIST 분류 참조 구현 스크립트](./scripts/mnist_classification.py)

## 주의사항

- PyTorch 백엔드에서 모델 출력 및 내부 연산은 `torch.Tensor` 타입입니다.
- torchvision `datasets.MNIST`는 `(1, 28, 28)` 채널-우선(CHW) 형태, Keras 기본 입력은 `(28, 28, 1)` 채널-마지막(HWC) 형태이므로 `transforms.Lambda(lambda x: x.permute(1, 2, 0))` 등으로 형태를 일치시켜야 합니다.

## Evals (발동 검증 질문)

- **Positive Triggers (이 스킬이 활성화되어야 함)**:
  - "Keras 3에서 PyTorch 백엔드로 Sequential 모델 만들고 컴파일해줘"
  - "PyTorch DataLoader를 Keras model.fit에 직접 넘기는 방법 알려줘"
  - "Keras로 MNIST 이미지 분류 파이프라인 컨벤션에 맞게 작성해줘"
  - "KERAS_BACKEND=torch 설정법과 주의점 알려줘"
- **Negative Triggers (이 스킬이 활성화되지 않아야 함)**:
  - "순수 PyTorch `nn.Module`과 학습 루프로 모델 작성해줘" ➔ 순수 PyTorch 안내
  - "Jena 기후 데이터로 시계열 LSTM 예측 모델 짜줘" ➔ `timeseries` 스킬 사용
  - "시스템의 conda 환경 전환해줘" ➔ `pydeep` 스킬 사용
