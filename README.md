# cifar10_resnet
간단한 이미지(숫자) 분류 모델입니다.

### Introduction

Deep Residual Learning for Image Recognition 논문을 읽고 section 4.2의 resnet model을 직접 구현해보았습니다.
학습 데이터셋은 cifar10을 사용했습니다.

### Requirements
After cloning the repo, run this line below:
```
pip install -r requirements.txt
```

### Usage

##### 1. train & test model
```
mkdir model
python -m cifar10_resnet.train -model {모델}
python -m cifar10_resnet.test -model {모델}
```

(추가예정)