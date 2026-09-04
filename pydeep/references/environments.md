# 환경별 패키지 상세

## conda 환경 목록

```bash
conda env list
```

```
# conda environments 예시:
#
# * -> active
base                 *   <conda_root>
pyml                     <conda_root>/envs/pyml
pytorch                  <conda_root>/envs/pytorch
tensorflow               <conda_root>/envs/tensorflow
```

## base

- Python: 3.12.x
- conda 기본 환경
- 딥러닝 프레임워크 없음

## pyml

- Python: 3.10.x
- 일반 파이썬 / 머신러닝용
- 주요 패키지:
  - numpy
  - pandas
  - matplotlib

## pytorch (권장 - Keras + PyTorch)

- Python: 3.10.x
- **Keras + PyTorch 백엔드** 실습용
- 패키지:
  - keras 3.x
  - torch 2.x
  - torchvision
  - numpy, pandas, matplotlib
- 가속기 지원: CUDA (Linux) 또는 MPS (macOS Apple Silicon)

### Keras 백엔드 확인

```bash
conda activate pytorch
python -c "import keras; print(keras.__version__, keras.backend.backend())"
# 예: 3.10.0 torch
```

### PyTorch 가속기 (CUDA / MPS) 확인

```bash
conda activate pytorch
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'MPS:', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())"
```

## tensorflow

- Python: 3.10.x
- TensorFlow 실습용
- 패키지:
  - tensorflow 2.x
  - numpy, pandas, matplotlib

### TensorFlow 버전 확인

```bash
conda activate tensorflow
python -c "import tensorflow as tf; print(tf.__version__)"
```

## 환경 활성화 요약

| 작업 | 명령 |
|------|------|
| conda 초기화 | `source $(conda info --base)/etc/profile.d/conda.sh` |
| Keras(PyTorch 백엔드) | `conda activate pytorch` |
| PyTorch | `conda activate pytorch` |
| TensorFlow | `conda activate tensorflow` |
| 일반 ML | `conda activate pyml` |
| 환경 목록 | `conda env list` |
