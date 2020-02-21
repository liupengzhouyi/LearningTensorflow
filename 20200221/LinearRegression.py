import tensorflow
import numpy
import matplotlib.pyplot as plt

X = numpy.random.uniform(3, 10, 100).astype(numpy.float32)
Y = 0.68 * X + 5.0 + numpy.random.uniform(-0.5, 0.5, 100).astype(numpy.float32)

# plt.plot(X, Y, 'ro', label='data')
# plt.show()

n_samples = X.shape[0]

#  y = k * x + b

W = tensorflow.Variable(numpy.random.randn(), name='weight')
b = tensorflow.Variable(numpy.random.randn(), name='bias')

# 线性回归的函数
def linear_regression(x):
    return W * x + b

# 损失函数
def mean_square(y_pred, y_true):
    return tensorflow.reduce_sum(tensorflow.pow(y_pred-y_true,2)) / (2*n_samples)

optimazer = tensorflow.optimizers.SGD(0.02)

def run_optimazation():
    with tensorflow.GradientTape() as g:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
    # 计算梯度
    gradients = g.gradient(loss, [W, b])
    # update Weight and bait
    optimazer.apply_gradients(zip(gradients, [W, b]))

for step in range(1, 10000):
    run_optimazation()
    if step % 500 == 0:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        print('step: %i, loss: %f, W: %f, b: %f' % (step, loss, W.numpy(), b.numpy()))
        plt.plot(X, Y, 'ro', label='data')
        plt.plot(X, numpy.array(W * X + b), label='fitted data')
        plt.legend()
        plt.show()




