# Simple Linear Regression

## Regression

- regression toward the mean : 전체의 평균으로 되돌아간다.
- 어떤 데이터들이 나와도 결과적으로 전체적으로 봤을때 전체 평균으로 되돌아가려는 속성이 있다.

## Linear Regression

- data를 가장 잘 대변하는 1차 방정식을 찾는 것
ex) y = ax + b

## Hypothesis

- H(x) = Wx + b
- Data를 가장 잘 대변하는 식
- 가설, 예측 이라고도 함

## Better Hypothesis

- 직선이 data를 가장 잘 대변하도록 찾는 것

## Cost, Cost function

- Cost: Hypothesis가 data의 차이로 작을수록 데이터를 잘 대변하고 있음을 알 수 있다.
- Cost, Loss, Error라고도 함

- sum(pow(hypothesis-real data)): 오차 제곱의 합
- Hypothesis: 기울기와 절편으로 구성된 직선 방정식
- Cost: 예측과 실제값의 오차 제곱의 평균

## Goal: Minimize cost

- Cost를 줄여주는 Weight와 bias를 찾는 것

## Gradient descent algorithm

- 경사 하강법: Cost를 최소화하는 W와 b를 찾는 알고리즘
1. 최초에 추정을 통해 W와 b값을 지정 (0이나 random값)
2. W와 b값을 cost가 줄어들도록 지속적으로 update함 (기울기 값을 줄여서 cost가 최소화되는 방향)
3. 최저점 도달까지 반복

- 경사를 따라 내려가면서 최저점을 찾음

## Convex function

- Local minimum: 주변에서 상대적으로 기울기가 가장 낮은 지점(잠시 기울기가 0 이 되는 지점)
- Local minimum이 많이 존재하는 경우 Gradient descent algorithm으로 최저점을 찾기는 어려움
