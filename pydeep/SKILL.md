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

### 0. Pre-flight Check (현재 환경 빠른 진단)

터미널에서 5초 내에 현재 파이썬 환경과 딥러닝 프레임워크 가용성을 점검합니다:

```bash
python -c "import sys; print(f'Python: {sys.version.split()[0]}'); \
import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()}, MPS: {getattr(torch.backends, \"mps\", None) and torch.backends.mps.is_available()})'); \
try: import keras; print(f'Keras: {keras.__version__} (Backend: {keras.backend.backend()})')
except ImportError: print('Keras: Not installed'); \
try: import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')
except ImportError: print('TensorFlow: Not installed')"
```

### 1. conda 초기화 및 환경 활성화

셸 세션에서 `conda activate`가 작동하지 않을 때는 conda 프로필 스크립트를 로드합니다:

```bash
# 동적 경로 탐색 (권장)
CONDA_BASE=$(conda info --base 2>/dev/null || echo "${CONDA_PREFIX:-$HOME/miniforge3}")
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 환경 활성화
conda activate pytorch
```

> 💡 매번 `source`를 입력하지 않으려면 셸 설정 파일(`~/.bashrc` 또는 `~/.zshrc`)에 conda init 블록을 추가하세요.

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
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available(), 'MPS:', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())"
# Linux 예: 2.8.0+cu128 CUDA: True MPS: False
# macOS 예: 2.8.0 CUDA: False MPS: True
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
- 하드웨어 가속: Linux에서는 CUDA(`torch.cuda.is_available()`), macOS(Apple Silicon)에서는 MPS(`torch.backends.mps.is_available()`)를 확인하세요.

## Evals (발동 검증 질문)

- **Positive Triggers (이 스킬이 활성화되어야 함)**:
  - "conda 환경을 pytorch로 변경해줘"
  - "이 머신에 텐서플로우와 파이토치 어떤 버전이 설치되어 있는지 확인해줘"
  - "터미널에서 conda activate pytorch 명령어가 안 먹히는데 어떻게 초기화하지?"
- **Negative Triggers (이 스킬이 활성화되지 않아야 함)**:
  - "Keras 3로 이미지 분류 Sequential 모델 작성해줘" ➔ `keras` 스킬 사용
  - "시계열 데이터 슬라이딩 윈도우 생성해줘" ➔ `timeseries` 스킬 사용
