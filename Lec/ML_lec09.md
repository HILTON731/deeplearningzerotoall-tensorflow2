# Neural Nets for XOR

## 2 layer
```py
def neural_net(features):
    layer1 = tf.sigmoid(tf.matmul(features, W1) + b1)
    layer2 = tf.sigmoid(tf.matmul(features, W2) + b2)
    hypothesis = tf.sigmoid(tf.matmul(tf.concat([layer1, layer2], -1), W3) + b3)
    return hypothesis
```
## Vector
```py
def neural_net(features):
    layer = tf.sigmoid(tf.matmul(features, W1) + b1)
    hypothesis = tf.sigmoid(tf.matmul(layer, W2) + b2)
    return hypothesis
```
- 하나의 layer에 있는 logistic regression들을 하나의 vector로 표현하여 batch 방식으로 연산 수행

# Tensorboard

- data를 통해 모델을 만드는 과정에서 weight나 bias값을 시각화해주는 tool

```py
!pip install tensorboard
tensorboard --logdir=./logs/xor_logs

- Youcan navigate to http://127.0.0.1:6006
```

### Eager Execution
```py
writer = tf.contirb.summary.FileWriter("./logs/xor_logs")
with tf.contrib.summary.record_summaries_every_n_global_steps(1):
    tf.contrib.summary.scalar('loss', cost)
```
### Keras
```py
tb_hist = tf.keras.callbacks.TensorBoard(log_dir="./logs/xor_logs", histogram_freq=0, write_graph=True, write_images=True)
model.fit(x_data, y_data, epochs=5000, callbacks=[tb_hist])
```