# Application & Tips

## Learning rate

- Gradient와 Learning rate값을 통해 최적의 모델 값을 찾을 수 있음

- Hyper-parameter: 모델을 만들기 위한 설정 값<br>
얼만큼 최적화하여 모델을 생성하는지에 결정
```py
def grad(hypothesis, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(hypothesis, labels)
        return tape.gradient(loss_value, [W, b])
optimizer = tf.keras.optimizer.SGD(learning_rate=0.01)
optimizer.apply_gradients(grads_and_vars=zip(grads, [W,b]))
```

- Learning rate의 값에 따라 overshooting 발생 가능 (주로 0.01 or 3e-4 사용)
```py
optimizer = tf.keras.optimizer.SGD(learning_rate=0.01 or 3e-4)
```

- Cost 값이 줄어들면 좋은 예측이라 볼 수 있고 cost 값이 증가하면 잘못된 예측이라 볼 수 있다.

### Annealing the learning rate(Learning rate decay)

- Cost 값이 점점 줄어들다가 어느 순간부터 학습이 되지 않는 부분에서 learning_rate를 조절하는 것으로 학습을 지속시키는 것

1. Step decay: 각 step 별로 epoch 만큼 rate 값을 조절하는 것
2. Exponential decay
3. 1/t decay: 1/epoch
```py
# Tensorflow code
learning_rate = tf.train.exponential_decay(starter_leanring_rate, gobal_step, 1000, 0.96, staircase)
or
tf.train.inverse_time_decay
or
tf.train.natural_exp_decay
or
tf.train.piecewise_constant
or 
tf.train.polynomial_decay

# python code
def exponential_decay(epoch):
    starter_rate = 0.01
    k = 0.96
    exp_rate = starter_rate * exp(-k*t)
    return exp_rate
```

## Data preprocessing

### Feature Scaling

1. Standarization(Mean Distance)
- 편차 / 표준 편차 (평균으로부터의 거리)
```py
Standarization = (data - np.mean(data)) / sqrt(np.sum((data - np.mean(data))^2) / np.count(data))
```

2. Normalization
- 실제 x값을 최솟값으로 뺀 값 / 최대 최솟값의 차이 (0과 1 사이의 값으로 정규화가 됨)
```py
Normalization = (data - np.min(data, 0)) / (np.max(data, 0) - np.min(data, 0))
```

### Noisy Data

- 이상치 값들(평균적인 데이터와 비교해 튀는 값)
<br>보통 전처리시 지워줌
<p> NLP나 Image 처리에서도 이상치 값을 지워주는 전처리과정이 필요함
