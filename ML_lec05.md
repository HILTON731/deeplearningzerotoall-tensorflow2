# Logistic Regression

## classification

- Binary classification: 이진 분류
<br> 0, 1로 두가지 케이스로 출력을 나누는 것
- Logistic regression data: 정적인 데이터를 선형으로 구분 가능
- Linear regression data: 연속적인 데이터를 수치형으로 구분

## Hypothesis Representation

    x(input data) --> Linear function --> Logistic function --> Decision boundary --> Y (imply {0, 1})

## Sigmoid function(Logistic function)

    원하는 값을 sigmoid 함수를 통해서 1과 0의 값으로 이진분류가 가능하도록 변환 가능

- sigmoid: 1과 0을 구분해주는 함수 
<br> Hypothesis = tf.sigmoid(z) or tf.div(1., 1. + tf.exp(z))
- Decision boundary: sigmoid를 통해 값을 구분하기 위한 기준 
<br> Predicted = tf.cast(hypothesis > 0.5, dtype=tf.int32)

## Cost Function
- Weight를 최적의 파라미터로 만들어주는 함수
- 실제 원하는 모델의 값과 실제 값과의 차이를 줄이는 것으로 원하는 모델 값을 만들어내는 것
```
def loss_fn(hypothesis, labels):
    cost = -tf.reduce_mean(labels * tf.log(hypothesis) + (1 - labels) * tf.log(1 - hypothesis))
    return cost
```
- 원하는 최적의 모델이 생성될 시 cost = 0
- 원하는 모델과 차이가 클 경우 cost값이 큼

 ## Optimization

 - Cost function을 반복하여 0에 가장 가까운 값을 구하는 것
 - tensorflow v2.0에서는 eager mode 지원
 ```
 def grad(hypothesis, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(hypothesis, labels)
    return tape.gradient(loss_value, [W, b])
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer.apply_gradients(grads_and_vars=zip(grads, [w, b]))
```

