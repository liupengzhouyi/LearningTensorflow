import tensorflow
import numpy
import matplotlib.pyplot

# 获取数据源
fashion_mnist = tensorflow.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255
test_images = test_images / 255

model = tensorflow.keras.Sequential()

model.add(tensorflow.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tensorflow.keras.layers.Dense(units=128, activation='relu'))
model.add(tensorflow.keras.layers.Dense(units=10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

model.fit(train_images, train_labels, epochs=10)





