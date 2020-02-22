import tensorflow

mnist = tensorflow.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test/255.0

train_ds = tensorflow.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tensorflow.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class LpModel(tensorflow.keras.Model):
    def __init__(self):
        super(LpModel, self).__init__()
        # self.conv1 = tensorflow.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten1 = tensorflow.keras.layers.Flatten(input_shape=(28, 28))
        self.d0 = tensorflow.keras.layers.Dense(units=28*28)
        self.flatten2 = tensorflow.keras.layers.Flatten(input_shape=(28, 28))
        self.d1 = tensorflow.keras.layers.Dense(units=128, activation='relu')
        self.d2 = tensorflow.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        inputs = self.flatten1(inputs)
        # inputs = self.conv1(inputs)
        inputs = self.d0(inputs)
        inputs = self.flatten2(inputs)
        inputs = self.d1(inputs)
        return self.d2(inputs)

model = LpModel()

loss_object = tensorflow.keras.losses.SparseCategoricalCrossentropy()

optimizer = tensorflow.keras.optimizers.Adam()

train_loss = tensorflow.keras.metrics.Mean(name='train_loss')
train_accuracy = tensorflow.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tensorflow.keras.metrics.Mean(name='test_loss')
test_accuracy = tensorflow.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tensorflow.function
def train_step(images, labels):
    with tensorflow.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    # 计算梯度
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

@tensorflow.function
def test_step(images, labels):
    predictions = model(images)
    loss = loss_object(labels, predictions)
    test_loss(loss)
    test_accuracy(labels, predictions)

EPOCHS = 5
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