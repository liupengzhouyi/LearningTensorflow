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

train_ds = tensorflow.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
test_ds = tensorflow.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

class LpClothesNet(tensorflow.keras.Model):
    def __init__(self):
        super(LpClothesNet, self).__init__()
        self.flatten1 = tensorflow.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tensorflow.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tensorflow.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        inputs = self.flatten1(inputs)
        inputs = self.dense1(inputs)
        return self.dense2(inputs)

model = LpClothesNet()

loss_function = tensorflow.keras.losses.SparseCategoricalCrossentropy()

optimizer = tensorflow.keras.optimizers.Adam()

train_loss = tensorflow.keras.metrics.Mean(name='train_loss')
train_accuracy = tensorflow.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tensorflow.keras.metrics.Mean(name='test_loss')
test_accuracy = tensorflow.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tensorflow.function
def train_step(images, labels):
    with tensorflow.GradientTape() as tape:
        predictions = model(images)
        loss = loss_function(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    # 计算梯度
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

@tensorflow.function
def test_step(images, labels):
    predictions = model(images)
    loss = loss_function(labels, predictions)
    test_loss(loss)
    test_accuracy(labels, predictions)

EPOCHS = 30
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = "Epoch: {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}."
    print(template.format(epoch+1, train_loss.result(), train_accuracy.result(), test_loss.result(), test_accuracy.result()))





