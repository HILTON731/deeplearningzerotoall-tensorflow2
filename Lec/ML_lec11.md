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

