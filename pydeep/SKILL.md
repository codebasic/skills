---
name: pydeep
description: '이 워크스페이스(파이딥 실습)의 시스템 conda 환경 사용법을 안내하는 스킬. base/pyml/pytorch/tensorflow 환경별 설치 패키지, 활성화 방법, Keras/PyTorch/TensorFlow 실행 방법을 다룹니다. Use when: conda 환경 활성화, conda activate, pytorch 환경, tensorflow 환경, pyml 환경, KERAS_BACKEND 설정, 파이토치/텐서플로우/케라스 실행, 환경 확인.'
argument-hint: 'conda 환경 사용법'
user-invocable: true
---

# 파이딥 conda 환경 사용법

이 워크스페이스는 파이딥(pydeep) 실습 환경으로, 시스템에 여러 개의 conda 환경이 설치되어 있습니다. 각 환경은 용도에 따라 다른 딥러닝 프레임워크를 포함합니다.

## When to Use

- conda 환경을 활성화하거나 전환해야 할 때
- Keras / PyTorch / TensorFlow 중 어떤 환경에서 실행할지 결정할 때
- 특정 프레임워크의 설치 여부나 버전을 확인할 때
- `conda activate` 사용법이 필요할 때

## conda 환경 목록

| 환경 | Python | 주요 패키지 | 용도 |
|------|--------|-------------|------|
| `base` | 3.12.12 | (기본) | conda 기본 환경 |
| `pyml` | 3.10.19 | numpy, pandas, matplotlib | 일반 파이썬/ML |
| `pytorch` | 3.10.19 | keras 3.10.0, torch 2.8.0+cu128, torchvision, numpy, pandas, matplotlib | **Keras(PyTorch 백엔드) + PyTorch** |
| `tensorflow` | 3.10.19 | tensorflow 2.20.0, numpy, pandas, matplotlib | TensorFlow |

## Procedure

### 1. conda 초기화 및 환경 활성화

터미널에서 conda를 사용하려면 먼저 conda를 초기화해야 합니다 (셸 세션마다 필요할 수 있음):

```bash
source /home/me/miniforge3/etc/profile.d/conda.sh
conda activate pytorch
```

> 💡 매번 `source`를 입력하지 않으려면 `~/.bashrc`에 위 `source` 줄을 추가하세요.

### 2. 환경 목록 확인

```bash
conda env list
```

### 3. 환경별 프레임워크 실행

**Keras (PyTorch 백엔드) - `pytorch` 환경:**
```bash
conda activate pytorch
python -c "import keras; print(keras.__version__, keras.backend.backend())"
# 예: 3.10.0 torch
```

**PyTorch - `pytorch` 환경:**
```bash
conda activate pytorch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 예: 2.8.0+cu128 True
```

**TensorFlow - `tensorflow` 환경:**
```bash
conda activate tensorflow
python -c "import tensorflow as tf; print(tf.__version__)"
# 예: 2.20.0
```

### 4. Keras 백엔드 설정

`pytorch` 환경의 Keras는 기본적으로 PyTorch 백엔드를 사용합니다. 명시적으로 설정하려면:

```bash
export KERAS_BACKEND="torch"
```

또는 코드에서 (Keras import 전에):
```python
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
```

## 참고 자료

- [환경별 패키지 상세](./references/environments.md)
- [Keras + PyTorch 백엔드 스킬](../keras/SKILL.md)

## 주의사항

- `base` 환경에는 딥러닝 프레임워크가 설치되어 있지 않습니다. 반드시 `pytorch` 또는 `tensorflow` 환경을 사용하세요.
- Keras 백엔드는 import 전에 설정해야 하며, 이후 변경 불가
- GPU(CUDA)는 `pytorch` 환경에서 사용 가능 (`torch.cuda.is_available()` → `True`)
