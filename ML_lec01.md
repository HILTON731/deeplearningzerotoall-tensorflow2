# Machine Learning?

일종의 software로 입력을 기반으로 출력을 하는 explicit programming을 하지 못하는<br>
logic이 많아 programming이 힘든 field를 자료를 통해 학습해서 배우는 명령

# Supervised learning vs Unsupervised learning

## Supervised learning
- Label이 정해져있는 data를 가지고 train 하는 것

ex) image labeling(사람들이 제공한 image를 통해 학습), 
Email spam filter, Predicting exam score

1. Machine learning이란 model이 있고 label이 정해진 data로 training을 한다.
2. Training을 통해 model 생성
3. Test data를 넣었을때(학습되지 않은 데이터를 넣었을때) 값을 예측해 주는 것
- Supervised learning의 기본 순서가 된다.

### Type of supervised learning

#### Regression
- Predicting final exam score based on time spent
(범위가 주어진 prediction)
#### Binary classification
- Pass/non-pass based on time spent
(Pass or Fail)
#### Multi-label classification
- Letter grade(A, B, C, E and F) based on time spent

## Unsupervised learning
- Data를 보고 스스로 train 하는 것

ex) Google news(자동적으로 grouping), World clustering

