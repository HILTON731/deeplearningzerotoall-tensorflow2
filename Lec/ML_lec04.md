# Multi variable linear regression

## Recap

- Hypothesis: H(x) = W * x + b (우리의 예측)
- Cost function: cost(W) = tf.reduce_mean(tf.square(H(x) - y)) (예측과 실제 값의 차이)
<br> 머신러닝의 핵심: cost를 최소화하는 W, b를 찾는 과정
- Gradient descent: W = W - a * reduce_mean(x * (H(x) - y))

## Prediction power

- 하나의 variable이 아닌 multi variable을 이용해서 learning을 수행하는 것
- H(x1, x2, x3) = w1 * x1 + w2 * x2 + w3 * x3 + b
- 변수 개수에 따라 가중치 값도 증가함
- metrix 사용: 행렬 곱(multi product)을 사용해서 표현
- H(X) = X * W

### metrix를 통한 장점
    1. 데이터 개수에 무관
    2. 많은 데이터를 동일한 연산식으로 정의 가능
    3. Weight의 개수는 입력 데이터와 출력 데이터의 형식을 통해 유추 가능




