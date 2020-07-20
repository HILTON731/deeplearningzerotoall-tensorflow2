# Relu

## Neural Network의 동작 방식

- Loss: 받은 input에 대한 output과 실제값(ground-truth)과의 차이
- Gradient: Loss를 미분하여 Back propagation하면서 네트워크(모델)를 학습시키는데 back propagation하면서 전달된 미분값 (그래프 상에서의 기울기)
- Sigmoid: Sigmoid 그래프의 가운데 부분의 기울기는 0보다 큼을 알 수 있지만 극단 좌표계(양 끝 부분)에서의 기울기는 0에 매우 가까움을 알 수 있다. Neural Network는 학습을 할때 Gradient를 전달받아 학습을 하는데 Gradient값이 너무 작으면 안됨

## Simoid 문제점

- 값을 전달할때마다 1이하의 값을 전달하기때문에 누적이 되다보면 0에 가까워져 처음 입력값의 의미가 없어지는 문제가 생김
= Vanishing gradient

## Solution

### Relu
```py
f(x) = max(0, x)
```
- 0 이하의 값은 0으로 값을 전달하지 않고(Relu의 문제점) 0이 아닌 양수값일 경우 y = x 식에 맞추어 값을 그대로 전달해주는 함수

### Leaky Relu
- 음수 부분에서의 값을 일부 해결
```py
tf.keras.layers # 내부에 있는 함수
```
- 0보다 작은 값을 가질 경우 값에 `a(알파)를 곱한 값을 반환 이외 Relu와 동일

### Else
- tanh, elu, selu ...etc
```py
tf.keras.activations # 내부에 있는 함수
```

## Xavier Initialization

- 실제 loss그래프는 간단하지 않고 여러 Local Minima가 존재 --> 시작점을 잘 설정해야 함
- 평균 = 0, 분산 = 2 / (Channel_in + Channel_out)<br>
Channel_in: Input으로 들어가는 channel의 개수<br>
Channel_out: Output으로 들어가는 channel의 개수

## He Initialization

- Relu함수에 특화된 Weight 초기화 방법
- 평균 = 0, 분산 = 4 / (Channel_in + Channel_out)

## Drop out

형성된 Neural Network에서 학습을 위해 모든 노드를 사용하는 것이 아니라 은닉층 별로 일부 노드들만 사용하여 학습을 진행시키는 것

- 사용하는 노드들은 임의로 설정
- Regularization: 정보의 모든 부분을 사용하는 것이 아닌 일부 부분들의 정보로 분해하여 학습을 진행하는 것이기에
- Test시에는 모든 노드를 활용

## Batch Normalization

- Internal Covariate Shift: NN이 학습을 진행할 때 layer를 지나면서 distribution이 변형되는 현상 발생.
```py
def batch_norm():
    return tf.keras.layers.BatchNormalization()
```
