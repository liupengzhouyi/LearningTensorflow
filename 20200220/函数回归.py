import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 一些参数
learning_rate = 0.02  # 学习率
training_steps = 2000  # 训练次数
display_step = 100  # 训练50次输出一次

# 训练数据
X = np.random.uniform(3, 10, 100).astype(np.float32)
Y = 0.3 * X + 0.7 + np.random.uniform(-0.5, 0.5, 100).astype(np.float32)

plt.plot(X, Y, 'ro', label='Original data')
plt.show()

n_samples = X.shape[0]

# 随机初始化权重和偏置
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# 线性回归函数
def linear_regression(x):
    return W*x + b

# 损失函数
def mean_square(y_pred, y_true):
    return tf.reduce_sum(tf.pow(y_pred - y_true, 2)) / (2 * n_samples)

# 优化器采用随机梯度下降(SGD)
optimizer = tf.optimizers.SGD(learning_rate)

# 计算梯度，更新参数
def run_optimization():
    # tf.GradientTape()梯度带，可以查看每一次epoch的参数值
    with tf.GradientTape() as g:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
    # 计算梯度
    gradients = g.gradient(loss, [W, b])
    # 更新W，b
    optimizer.apply_gradients(zip(gradients, [W, b]))

# 开始训练
for step in range(1, training_steps+1):
    run_optimization()
    if step % display_step == 0:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy()))
        plt.plot(X, Y, 'ro', label='Original data')
        plt.plot(X, np.array(W * X + b), label='Fitted line')
        plt.xlabel('fitted linear regression')
        plt.legend()
        plt.show()



