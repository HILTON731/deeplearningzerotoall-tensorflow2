# Softmax Regression

## Logistic regression(Sigmoid)

    return 형태가 실수이기 때문에 binary classification에 부적합
- z = H(x)라는 return 값이 있고 g(z)라는 함수를 통해 실수값을 압축하여 0 ~ 1 값으로 표현하도록 하는 것
- g(z) = 1 / (1 + exp(-2)) [sigmoid / logistic regression]
- g(H(x)) = Y`

학습시킨다 : 데이터 군을 구분하는 선을 찾는 것 (hyperplane을 찾는다.)

## Multinomial classification

    binary classification을 통해 구현 가능
    (A, B, C에 대해 각각 classifier를 가지도록 구현)
    --> 세번의 독립된 형태의 vector를 가지고 계산하도록 함

행렬을 통해 사용할경우(Matrix multiplication)
    
    [[wa1, wa2, wa3],
    [wb1, wb2, wb3],
    [wc1, wc2, wc3]]
    이렇게 하나의 vector를 통해 독립된 classification 구현 가능

## Softmax
    S(Y(i)) = exp(y(i)) / sum(exp(y(j)))
```
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
tf.matmul(X, W) + b
```
- 여러 개의 class를 예측할때 유용
- 출력값을 확률 값으로 변환
- 입력에 대한 vector값을 0과 1사이의 값으로 표현
- 각 vector값의 총 합은 1
- One-hot encoding을 통해 각 vector값을 확률로 간주하여 가장 높은 값을 하나의 값으로 표현

## Softmax에서의 cost function

- Cross-Entropy를 통해 실제 값과 예측 값의 차이를 비교
- -sum(L(i) * log(Y'(i))) L(i): 실제 값, Y'(i): 추정 값
- log(Y'(i)): 0 ~ 1 값이 됨



