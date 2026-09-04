# 설치 및 백엔드 설정

## conda pytorch 환경 (이 워크스페이스)

이 워크스페이스에서는 시스템의 conda `pytorch` 환경을 사용합니다. Keras 3.10.0과 PyTorch 2.8.0(CUDA 지원)이 이미 설치되어 있습니다.

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

## Keras 3 설치

Keras 3는 PyPI에서 설치할 수 있습니다:

```bash
pip install --upgrade keras
```

Keras 3를 사용하려면 백엔드 프레임워크(JAX, TensorFlow, PyTorch 중 하나)도 설치해야 합니다. PyTorch 백엔드를 사용하려면:

```bash
pip install torch torchvision
```

### 버전 호환성

| Keras | PyTorch |
|-------|---------|
| `keras~=3.0` | `torch~=2.1.0` |

### KerasCV / KerasHub (선택)

컴퓨터 비전 / NLP 확장 모듈:

```bash
pip install --upgrade keras-cv
pip install --upgrade keras-hub
```

## 백엔드 설정

백엔드는 **Keras를 import 하기 전에** 설정해야 합니다. 설정 후에는 변경할 수 없습니다.

### 방법 1: 환경 변수

```bash
export KERAS_BACKEND="torch"
```

### 방법 2: 코드 내 설정 (노트북에서 권장)

```python
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
```

### 방법 3: 설정 파일

`~/.keras/keras.json` 파일을 편집:

```json
{
    "backend": "torch"
}
```

사용 가능한 백엔드: `"jax"`, `"tensorflow"`, `"torch"`

## 백엔드 확인

```python
import keras
print(keras.__version__)          # 예: 3.x.x
print(keras.backend.backend())    # 예: 'torch'
```

## GPU 사용

PyTorch 백엔드에서 GPU를 사용하려면 CUDA 지원 PyTorch가 설치되어 있어야 합니다.

```python
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU name:', torch.cuda.get_device_name(0))
```

Keras는 자동으로 사용 가능한 GPU를 감지하여 사용합니다.

## 주의사항

- `tensorflow==2.15`를 설치하면 Keras 2로 덮어쓰므로, Keras 3를 다시 설치해야 합니다. (TensorFlow 2.16부터는 Keras 3가 기본)
- 백엔드는 import 전에 설정해야 하며, 이후 변경 불가
- PyTorch 백엔드에서는 텐서가 `torch.Tensor` 타입으로 표현됨
