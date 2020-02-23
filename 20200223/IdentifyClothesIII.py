import numpy
import tensorflow
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 获取数据源
fashion_mnist = tensorflow.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255
test_images = test_images / 255

print(test_labels[0])

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

model.fit(train_images, train_labels, epochs=1)

property = model.predict(test_images)

plt.figure(figsize=(3, 9))

def pp(i, property):
    plt.subplot(5,2,2*i)
    plt.xticks(range(10))
    plt.bar(range(10), property, color='red')
    plt.ylim([0, 1])
    plt.xlabel('image01')

def paly(i, index):
    plt.subplot(5, 2, (i-1)*2 + 1)
    propertys = property[index]
    number = numpy.argmax(propertys)
    class_name = class_names[number]
    plt.imshow(test_images[index])
    plt.xlabel(class_name)
    plt.grid(False)
    pp(i, propertys)


paly(1, 101)
paly(2, 108)
paly(3, 102)
paly(4, 151)
paly(5, 191)



plt.show()
