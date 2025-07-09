# Title (Please modify the title)

## Team

| ![김시진](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김영]!<img src="./project1/me.jpg" alt="내 사진" width="140px"> | ![정예은](https://avatars.githubusercontent.com/u/156163982?v=4) | ![전수정](https://avatars.githubusercontent.com/u/156163982?v=4) | 
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김시진](https://github.com/UpstageAILab)             |            [김영](https://github.com/kimyoung9689)             |            [정예은](https://github.com/UpstageAILab)             |            [전수정](https://github.com/UpstageAILab)             |            
                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 1. Competiton Info

### Overview

- _Write competition information_

### Timeline

- ex) January 10, 2024 - Start Date
- ex) February 10, 2024 - Final submission deadline

### Evaluation

- Evaluation (평가 방식)
모델의 성능은 [평가 지표, 예: RMSE (Root Mean Squared Error) for Regression, Accuracy / F1-score for Classification]를 기준으로 평가되었습니다. [평가 지표가 의미하는 바를 간략히 설명, 예: "RMSE는 예측 오차의 크기를 나타내며, 값이 낮을수록 모델의 예측 정확도가 높음을 의미합니다."]

## 2. Components

### Directory

- _Insert your directory structure_

## 3. Data descrption

### Dataset overview

- 본 프로젝트에서 사용된 데이터셋은 [데이터셋 이름, 예: '서울시 아파트 실거래가 데이터']입니다. 데이터는 총 [데이터셋 크기, 예: 10,000개]의 행과 [컬럼 수, 예: 15개]의 컬럼으로 구성되어 있으며, [데이터의 주요 특징, 예: '부동산 거래일, 면적, 층수, 지역 정보, 거래 가격' 등의 정보를 포함합니다.].

### EDA

- 데이터의 특성을 이해하기 위해 다음과 같은 탐색적 데이터 분석(EDA)을 수행했습니다.

데이터 분포 확인: 각 컬럼의 값 분포, 결측치 여부, 이상치 등을 시각화하여 확인했습니다.

변수 간 상관관계 분석: 주요 변수들 간의 상관관계를 파악하여 모델링에 활용할 인사이트를 도출했습니다.

시계열/지역별 트렌드 분석: [데이터 특성에 맞는 추가 분석, 예: '시간에 따른 가격 변화 추이', '지역별 가격 편차' 등을 분석했습니다.]

주요 결론: [EDA를 통해 얻은 핵심적인 결론 2~3가지 요약, 예: "특정 지역의 가격 편차가 크고, 거래일에 따른 계절성이 관찰됨."]

### Feature engineering

- 모델 성능 향상을 위해 다음과 같은 특징 공학(Feature Engineering) 작업을 수행했습니다.

파생 변수 생성: [원래 데이터에 없던 새로운 변수를 만든 예시, 예: '거래일에서 연도, 월, 요일 정보 추출', '면적당 가격 비율 계산']

범주형 변수 인코딩: [범주형 데이터를 숫자로 변환한 방법, 예: '원-핫 인코딩(One-Hot Encoding)' 또는 '레이블 인코딩(Label Encoding)']

스케일링: [데이터 스케일을 조정한 방법, 예: '수치형 변수들을 표준화(Standardization) 또는 정규화(Normalization)하여 모델 학습에 적합하도록 변환']

## 4. Modeling

### Model descrition

- 본 프로젝트에서는 [문제 유형, 예: '회귀 문제' 또는 '분류 문제'] 해결을 위해 [선택한 모델 이름, 예: 'LightGBM', 'XGBoost', 'Random Forest', 'LSTM' 등]을 주 모델로 선정했습니다.

선정 이유: [해당 모델을 선택한 이유, 예: "LightGBM은 대용량 데이터 처리 및 빠른 학습 속도, 높은 예측 성능으로 정형 데이터에 강점을 보이기 때문입니다." 또는 "LSTM은 시계열 데이터의 장기 의존성 학습에 유리하여 시계열 예측에 적합하다고 판단했습니다."]

하이퍼파라미터 튜닝: [하이퍼파라미터 튜닝 방식, 예: 'Grid Search', 'Random Search', 'Optuna' 등을 활용하여 최적의 하이퍼파라미터를 탐색했습니다.]

### Modeling Process

- 모델 학습 및 평가 과정은 다음과 같습니다.

데이터 분할: 전체 데이터를 학습(Train) 세트와 검증(Validation) 세트, 테스트(Test) 세트로 분할했습니다.

모델 학습: 학습 세트를 사용하여 모델을 훈련시켰습니다.

성능 평가: 검증 세트를 사용하여 모델의 중간 성능을 평가하고 하이퍼파라미터 튜닝에 활용했습니다.

최종 예측: 최종 선정된 모델로 테스트 세트에 대한 예측을 수행했습니다.

## 5. Result

### Leader Board

평가 지표: RMSE
- 1일차 46707.2744
- 2일차 43015.0267

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference
- 서울시 공동주택 아파트 정보
- [_Insert related reference_](https://data.seoul.go.kr/dataList/OA-15818/S/1/datasetView.do)
