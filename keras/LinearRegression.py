import tensorflow.keras
import numpy
import matplotlib.pyplot as plt

X = numpy.random.uniform(3, 10, 100).astype(numpy.float32)
y = X * 0.35 + 5.0 + numpy.random.uniform(-0.5, 0.5, 100).astype(numpy.float32)
model = tensorflow.keras.Sequential()
model.add(tensorflow.keras.layers.Dense(input_dim=1, units=1))
model.compile(loss='mse', optimizer='sgd')

for step in range(0, 1000):
    cost = model.train_on_batch(X, y)
    if step % 45 == 0:
        print('step: %d, cost: %f' % (step, cost))
        plt.plot(X, y, 'ro', label='data')
        plt.plot(X, model.predict(X), label='fitted data')
        plt.legend()
        plt.show()

