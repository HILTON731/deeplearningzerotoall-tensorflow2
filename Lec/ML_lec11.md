# CNN(Convolutional Neural Network)

- Image classification에서 가장 많이 사용되고 있는 분류 알고리즘
- 특징 추출: Convolution, Pooling layer
- 분류: Fully-Connected

```py
# N * N * 3 Image가 있다고 가정
# F * F Filter로 통과시킬 경우
(N - F) / stride + 1
```
학습이 진행됨에 따라 loss가 생김을 알 수 있음

## Padding

이미지의 테두리에 0값을 씌우는 것
- 이미지가 급격하게 작아지는 것 방지
- 이미지의 끝점 표시

## Filter

- 여러개의 필터를 사용한 값을 나타냄
- Filter 하나당 나오는 값: Feature map, Activation map
```py
(_, _, Filter_num)
```

## Pooling

### Max Pooling vs Average Pooling

#### Max Pooling(Sub Sampling):
 Filter를 통과한 값중 가장 큰 값을 대표값으로 사용하는 것

#### Average Pooling: 
Filter를 통과한 값의 평균값을 대표값으로 사용하는 것

- Max Pooling을 더 자주 사용: Conv연산을 통해 나온 연산 결과중 큰 값은 찾으려는 특징에 가깝다는 의미이기 때문

# RNN(Recurrent Neural Network)

- Sequential data에 적합한 NN모델

```py
h(t) = fw(h(t-1), x(t))
```

- 이전 입력값을 예측모델에 일부 반영하여 예측 수행

## RNN 구현

```py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# version 1
cell = layers.SimpleRNNCell(units=hidden_size)
rnn = layers.RNN(cell, return_sequences=True, return_state=True)
outputs, states = rnn(x_data)

# version 2
rnn = layers.SimpleRNN(units=hidden_size, return_sequences=True, return_state=True)
outputs, states = rnn(x_data)
```
### version 1
- 특정 cell을 선언하고 이를 loop하는 방식

### version 2
- 두가지 방식 합쳐서 사용

## RNN input shape
```py
shape=[batch_size, sequence_len, input_dim]
```

## Various usage of RNN

1. one to one: 하나의 입력을 받아 하나의 값을 출력
2. one to many: 특정 입력을 받아 cation을 생성해주는 image captioning에 많이 사용
3. many to one: 자연어 처리에서 문장을 분류해주는 classification에서 사용
4. many to many: 문장을 입력을 받아 문장을 출력하는 language translation혹은 형태소를 분석해주는 형태소 분석기로 사용 가능

## Many to One

Tokenization: Sentence를 word의 sequence라고 생각하고 sentence를 word 단위로 분류<br>
RNN을 통해  결과값의 polarity 분석하는 방식
<p>
RNN은 word를 직접 처리할 수 없기때문에 token을 numeric vector로 바꿔주는 Embedding layer 존재
<br> 
활용 방식에 따라 학습을 할 수도 안할 수도 있음
</p>
RNN을 통해 token을 읽어서 나온 최종 값과 실제 값을 비교해 back propagation하여 training 진행

## Multi-layered RNN(Stacked RNN)
- CNN과 유사하게 RNN을 여러 층으로 쌓는 것
- 경험적 측면에서 성능이 shallow RNN (단층 RNN)에 비해 성능이 좋았음

### In NLP
Input에 가까운 hidden layer는 syntactic information을 더 잘 파악하고<br>
Output에 가까운 hidden layer는 semantic information을 더 잘 파악하는 경향이 있음

