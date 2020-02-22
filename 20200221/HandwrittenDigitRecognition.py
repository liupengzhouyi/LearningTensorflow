import tensorflow
import numpy
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

mnist = tensorflow.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test/255.0

model = tensorflow.keras.models.Sequential([
    # 展平输入
    tensorflow.keras.layers.Flatten(input_shape=(28, 28)),
    # 全联接层
    tensorflow.keras.layers.Dense(units=128, activation='relu'),
    # 减缓过拟合
    tensorflow.keras.layers.Dropout(0.2),
    # 输出
    tensorflow.keras.layers.Dense(units=10, activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5)

x = [0., ]
y = [1., ]

for step in range(1001):
    cost = model.train_on_batch(x_train, y_train)
    x.append(step)
    y.append(1.0 - cost[1])
    X = numpy.array(x)
    Y = numpy.array(y)
    if step % 10 == 0:
        print(step, ": ", cost)
        plt.plot(X, Y, color='red', label='error')
        plt.show()

model.evaluate(x_test, y_test, verbose=2)

