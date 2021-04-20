# P stage 2: Relation Extraction


```
$> tree -d
.
├── /dataloader
│     ├── create_fold.py: dataset을 나눈다.  
│     ├── load_data.py: info.csv에 미리 클래스를 만들어주고 이미지 파일을 jpg로 통일 
│     ├── dataset.py: Dataset Class 정의
├── /model
│     ├── loss.py: 다양한 loss class를 정의
│     ├── model.py: 다양한 model class을 정의
│     └── optimizer.py: 다양한 optimizer 반환 함수 정의
├── /trainer
│     ├── trainer.py: gender_age 모델 학습을 위한 Trainer class 정의
├── config.py: Hyperparameter 를 불러온다.
├── config.yml: Hyperparameter 저장
├── inference.py
├── train.py
└── utils.py : 그 외 필요한 기능
``` 

## 요약 

- 기간: 2021.03.29 ~ 2021.04.08
- 대회 설명: 문장에서 지정된 entity 사이에서 관계를 추출한다.
- 검증 전략: Stratified K-fold Cross Validation(n=5)
- 사용한 모델 아키텍처 및 하이퍼 파라미터
    - Bert for multilingual base
    - Optimizer: AdamW
    - Learning rate: 
    - Batch size: 1
    - Optuna를 이용해 hyperparameter search를 진행
<br/>

## Scores
|  |Public LB|Private LB|
|--|--|--|
|Accuracy|||

<br/>

## 도움이 되었던 것
1. small batch size 

<br/>

## 도움이 되지 않은 것
1. Entity를 special token으로 대체

<br/>

## Code
### Train
```bash
$ python train.py --model base
```
### Inference
```bash
$ python inference.py
```
