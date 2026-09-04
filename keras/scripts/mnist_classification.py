"""
MNIST 이미지 분류 - Keras 3 (PyTorch 백엔드) 참조 구현

노트북 컨벤션에 따라 작성된 예제:
    - 반복 작업(데이터로더 구성, 평가)은 유틸리티 함수로 추출
    - 설정(batch_size, num_workers, epochs 등)은 키워드 인자로 전달 (하드코딩 금지)
    - 모델 구축은 build_model() 형태의 함수로 추출
    - 성능 비교는 DataFrame으로 표시

사용법 (conda pytorch 환경):
    conda activate pytorch
    export KERAS_BACKEND="torch"
    python mnist_classification.py
"""
import os

# 백엔드는 Keras import 전에 설정해야 함
os.environ["KERAS_BACKEND"] = "torch"

import keras
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# 유틸리티 함수 (import 직후 정의)
# ---------------------------------------------------------------------------

def create_dataloaders(datasets_dict, **kwargs):
    """데이터로더를 구성합니다.

    Args:
        datasets_dict: 데이터셋 딕셔너리
        **kwargs: DataLoader에 전달할 키워드 인자 (batch_size, num_workers 등)

    Returns:
        데이터로더 딕셔너리
    """
    dataloaders = {}
    for split in datasets_dict:
        dataloaders[split] = DataLoader(
            datasets_dict[split],
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


def build_mlp(input_shape=(28, 28, 1), hidden_size=128, num_classes=10):
    """MLP 모델을 구축합니다."""
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(hidden_size, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def build_cnn(num_classes=10):
    """합성곱 모델을 구축합니다.

    합성곱 블록을 반복하며, 입력의 공간적 차원이 너무 작지 않은
    경우에만 풀링을 적용합니다.
    """
    inputs = keras.Input(shape=(28, 28, 1))
    x = inputs
    for filters in [32, 64, 128]:
        x = keras.layers.Conv2D(
            filters=filters, kernel_size=3, activation='relu')(x)
        # 입력의 공간적 차원이 너무 작지 않은 경우에만 풀링 적용
        if x.shape[1] > 3:
            x = keras.layers.MaxPooling2D(pool_size=2)(x)

    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def create_mnist_datasets(root='./data', transform=None):
    """MNIST 데이터셋 딕셔너리를 구성합니다."""
    return {
        split: datasets.MNIST(
            root=root, train=split == 'train', download=True,
            transform=transform)
        for split in ['train', 'test']
    }


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main(batch_size=100, num_workers=2, epochs=5):
    """MNIST 분류 파이프라인 전체를 실행합니다."""
    import pandas as pd

    # 전처리: (C, H, W) -> (H, W, C) 채널-마지막 형태로 변환
    전처리 = transforms.Compose([
        transforms.ToTensor(),  # PIL 이미지를 텐서로 변환
        # (C, H, W) -> (H, W, C)
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),
    ])

    # 데이터셋 및 데이터로더 구성 (설정은 키워드 인자로)
    mnist_datasets = create_mnist_datasets(transform=전처리)
    dataloader = create_dataloaders(
        mnist_datasets, batch_size=batch_size, num_workers=num_workers)

    # 배치 형태 확인
    for x_batch, y_batch in dataloader['train']:
        print(x_batch.shape, y_batch.shape)
        break

    # 두 모델(MLP vs 합성곱)을 학습하고 성능 비교
    histories = {}
    scores_by_name = {}
    for name, build_fn in [('MLP', build_mlp), ('CNN', build_cnn)]:
        model = build_fn()
        model.summary()
        model.compile(
            optimizer='rmsprop',
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
        )
        # 학습
        histories[name] = model.fit(dataloader['train'], epochs=epochs)
        # 평가
        scores_by_name[name] = evaluate_model(model, dataloader)

    # 성능 비교 표시
    print('\n=== 성능 비교 (test) ===')
    print(pd.DataFrame(
        {name: s['test'] for name, s in scores_by_name.items()}
    ).round(3))


if __name__ == '__main__':
    print(f'Keras version: {keras.__version__}')
    print(f'Backend: {keras.backend.backend()}')
    main(epochs=5)

