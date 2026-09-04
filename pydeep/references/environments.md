# 환경별 패키지 상세

## conda 환경 목록

```bash
conda env list
```

```
# conda environments:
#
# * -> active
# + -> frozen
base                 *   /home/me/miniforge3
pyml                     /home/me/miniforge3/envs/pyml
pytorch                  /home/me/miniforge3/envs/pytorch
tensorflow               /home/me/miniforge3/envs/tensorflow
```

## base

- Python: 3.12.12
- conda 기본 환경
- 딥러닝 프레임워크 없음

## pyml

- Python: 3.10.19
- 일반 파이썬 / 머신러닝용
- 패키지:
  - numpy 2.2.6
  - pandas 2.3.3
  - matplotlib 3.10.8

## pytorch (권장 - Keras + PyTorch)

- Python: 3.10.19
- **Keras + PyTorch 백엔드** 실습용
- 패키지:
  - keras 3.10.0
  - torch 2.8.0+cu128
  - torchvision 0.23.0+cu128
  - numpy 2.2.6
  - pandas 2.3.3
  - matplotlib 3.10.8
- GPU(CUDA) 사용 가능

### Keras 백엔드 확인

```bash
conda activate pytorch
python -c "import keras; print(keras.__version__, keras.backend.backend())"
# 3.10.0 torch
```

### PyTorch GPU 확인

```bash
conda activate pytorch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.8.0+cu128 True
```

## tensorflow

- Python: 3.10.19
- TensorFlow 실습용
- 패키지:
  - tensorflow 2.20.0
  - numpy 2.2.6
  - pandas 2.3.3
  - matplotlib 3.10.8

### TensorFlow 버전 확인

```bash
conda activate tensorflow
python -c "import tensorflow as tf; print(tf.__version__)"
# 2.20.0
```

## 환경 활성화 요약

| 작업 | 명령 |
|------|------|
| conda 초기화 | `source /home/me/miniforge3/etc/profile.d/conda.sh` |
| Keras(PyTorch 백엔드) | `conda activate pytorch` |
| PyTorch | `conda activate pytorch` |
| TensorFlow | `conda activate tensorflow` |
| 일반 ML | `conda activate pyml` |
| 환경 목록 | `conda env list` |
