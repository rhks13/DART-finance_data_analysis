# 국면인식을 이용한 시장 방향성 분석
## 배경

### ' 현재와 비슷한 과거는 미래에 대한 단서를 제공하는가?'

이에 답하기 위해 현재와 가장 비슷한 과거 국면을 매크로 변수를 이용해 정량적으로 선정하였으며,
선정된 과거 국면과 코스피 시장 수익률의 관계를 통해 시장의 방향성을 탐구함

## 선행연구
1. 현시점과 매크로환경이 유사한 과거 시점을 찾고
2. 유사한 과거시점에서 평균적으로 잘 맞았던 팩터 검색
3. 해당 기준으로 팩터와 팩터 가중치 결정
- 출처 : 삼성증권, 김동영, 매크로 기반 다이나믹 퀀트 모델

## 실험설계
### 1. 시점 X와 가장 비슷한 지점은 어디일까?
### 2. 매크로 변수 선정 -> correl-adj dist 근거, '시점 X'와 유사시점 후보 n개 선정 (표준화 거리 하위 n개)
  ![image](https://user-images.githubusercontent.com/69777594/231328562-869680d3-6996-411c-9a7d-204b2d337440.png)

   ![image](https://user-images.githubusercontent.com/69777594/231329196-203c022b-a157-46ba-b713-0d2e07bb08de.png)

### 3. 유사시점 후보 n개는 실제로 시점 x와 유사할까?

### 3-1. 시점 x의 코스피 수익률과 유사시점 후보 6개의 코스피 평균 수익률간 상관관계 계산
  ![image](https://user-images.githubusercontent.com/69777594/231329234-722ea5fa-88c3-4d24-987f-681ad1c284c6.png)

### 3-2. 시점 x의 BULL/BEAR 판단 결과와 유사시점의 판단결과 비교
  ![image](https://user-images.githubusercontent.com/69777594/231329291-db674247-ec84-45ef-9cf8-78e3f8a92aa2.png)

## 검증결과
  ![image](https://user-images.githubusercontent.com/69777594/231329437-66aec482-93b9-42eb-a50d-37030b71853c.png)
  ![image](https://user-images.githubusercontent.com/69777594/231329460-0611a53b-8676-4325-89c9-9704e1d0e1c9.png)

## 결론
+ 22년 4월 30일 기준 3개월 코스피의 향후 방향 -> BULL/BEAR가 아닌 횡보장일 것
+ 한계
  + 매크로 지표 선정 근거에 대한 논의 부족
  + 일간 매크로 지표 사용하지 못함
  + 세계시장 반영 매크로 지표의 사용 불충분
