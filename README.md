# 머신러닝을 이용한 팀리더와 지원자 추천 시스템
---
## 프로젝트 개요 :
---
이 프로젝트는 머신러닝 기법 (word2Vec, K-means clustering, PCA)을 활용하여 팀리더와 지원자간의 매칭을 돕는 추천시스템이다.
프로젝트에서는 팀 리더와 지원자의 "기술 스킬" , " 경력 데이터" 등을 기반으로 비슷한 프로필을 가진 사람들을 그룹화하여 최적의 매칭을 지원한다.
![image](https://github.com/user-attachments/assets/97ca01af-5969-47c0-b288-221774d77b2b)

![image](https://github.com/user-attachments/assets/55217a91-ffd1-4092-bc65-af49f9e8300f)

## 사용된 언어 & 라이브러리 및 기술 :
---
사용 언어 : Python

라이브러리 및 프레임워크 : 
1. Pandas : 데이터 로드 및 전처리
2. Numpy : 벡터화된 데이터 처리
3. Scikit-learn : 머신러닝 모델 및 데이터 전처리 파이프라인 구현
4. pytorch : 자연어처리 모델인 word2Vec을 훈련하여 문자열 데이터를 벡터화
5. matplotlib : 클러스터링 결과 시각화



## 프로젝트의 주요 단계 및 과정:
---
데이터 : 챗 지피티를 사용하여 "현업 분야를 고려하여 지원자와 팀 리더의 가상의 요구사항" 을 반영한 더미 데이터를 만들었다. 데이터 프레임에서 row는 각 사용자 혹은 팀 리더 ID를 의미하고, col은 지원자와 팀 리더 둘다 동일하지만, 바라보는 관점이 다른 데이터이다. 데이터 프레임의 size는 20000x12로 사용자와 리더의 정보가 통합된 데이터이다. 데이터 자체는 실제 웹 서비스를 진행하며 얻어지는 값과 피쳐의 값을 동일하게 하여 최대한 유사한 상황을 가상으로 만들어 학습하였다. 하지만 실제 데이터는 아니기 때문에, 이 부분은 직접 서비스를 출시해서 사용자 데이터가 얻어짐에 따라 모델의 개선할 부분이 분명 존재할 것이다.

동작 메커니즘 :
 + Pandas를 사용하여 CSV 파일로부터 데이터를 로드하고, 각 샘플의 ID를 인덱스로 설정하였다.
 + 수치형 데이터는 min-max 정규화, 범주형 데이터는 원-핫 인코딩, 텍스트 문자열 데이터에는 딥러닝 모델인 Word2Vec을 사용하였다.
 + 최종적으로 얻어진 수치데이터를 하나의 벡터로 통합하였다.
 + PCA를 사용하여 데이터를 2차원으로 축소한 후, K-means 클러스터링을 통해 데이터를 그룹화하였다.
 + 최적의 클러스터 수를 결정하기 위해 엘보우 방법(Elbow Method)을 사용하여 시각적으로 분석.
 + Matplotlib와 Seaborn을 사용하여 클러스터링 결과를 시각화하였다. 

결과 :
클러스터링된 데이터는 팀리더와 지원자 간의 매칭 시스템으로 활용될 수 있으며, 이 정보를 바탕으로 최적의 추천을 할 수 있다.



## 데이터 샘플 수 추가 해서 진행
---
현재는 팀리더와 지원자 100명에 대한 가상 데이터를 만들어서 구현을 목적으로 만들었고,
이제 팀리더와 지원자를 10000명으로 늘려서 다시 수행

![image](https://github.com/user-attachments/assets/280f3be7-c0ab-48fe-bbc3-c54d36fdcd0f)
![image](https://github.com/user-attachments/assets/83dad81b-ae5f-4888-8ffa-8cc65551096e)

여기서 A로 시작하는게 지원자 ID , T로 시작하는게 프로젝트 매니저 ID


## flask 서버 구현 ( spring 백엔드와 통신하는 머신러닝 모델 실행 서버)
---
Flask 파이썬 프레임워크를 사용해 머신러닝 모델을 실행하기 위한 별도의 서버를 구축하고, 스프링 백엔드와는 REST API를 사용해 통신

1. 클라이언트 A의 정보를 Json 데이터를 Falsk 라우트에서 처리
2. 받은 데이터를 df구조로 바꾼 후에, 전처리 및 모델 예측 수행.
3. 클라이언트 A와 같은 클러스터에 속한 팀 리더들의 ID를 리스트 형태로 반환 결과는 다시, Json 데이터
