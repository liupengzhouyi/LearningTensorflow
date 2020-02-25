import tensorflow
import matplotlib.pyplot
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

(train_data, train_labels), (test_data, test_labels) = tensorflow.keras.datasets.boston_housing.load_data()

print(train_data.shape)

print(train_labels.shape)

model = tensorflow.keras.Sequential()
model.add(
    tensorflow.keras.layers.Dense(
        units=32,
        activation=tensorflow.keras.activations.sigmoid,
        input_shape=(13,)
    )
)
model.add(
    tensorflow.keras.layers.Dense(
        units=64,
        activation=tensorflow.keras.activations.sigmoid
    )
)
model.add(
    tensorflow.keras.layers.Dense(
        units=32,
        activation=tensorflow.keras.activations.sigmoid,
    )
)
model.add(
    tensorflow.keras.layers.Dense(
        units=1
    )
)

model.compile(
    optimizer=tensorflow.keras.optimizers.SGD(
        learning_rate=0.1,
    ),
    loss=tensorflow.keras.losses.mean_squared_error,
    metrics=['mse']
)

model.summary()

history = model.fit(
    train_data,
    train_labels,
    batch_size=50,
    epochs=60,
    validation_split=0.1,
    verbose=1
)


result = model.evaluate(test_data, test_labels)

print(result)

train_result = history.history
length = [x for x in range(60)]
loss = train_result['loss']
val_loss = train_result['val_loss']
mse = train_result['mse']
val_mse = train_result['val_mse']

matplotlib.pyplot.subplot(2, 2, 1)
matplotlib.pyplot.plot(length, loss)
matplotlib.pyplot.title('loss')

matplotlib.pyplot.subplot(2, 2, 2)
matplotlib.pyplot.plot(length, val_loss)
matplotlib.pyplot.title('val_loss')

matplotlib.pyplot.subplot(2, 2, 3)
matplotlib.pyplot.plot(length, mse)
matplotlib.pyplot.title('mse')

matplotlib.pyplot.subplot(2, 2, 4)
matplotlib.pyplot.plot(length, val_mse)
matplotlib.pyplot.title('val_mse')
matplotlib.pyplot.savefig('image123.png')
matplotlib.pyplot.show()





matplotlib.pyplot.figure(figsize=(5, 40), dpi=300)

for i in range(12):
    list = train_data[:, i]
    matplotlib.pyplot.subplot(12, 1, i + 1)
    matplotlib.pyplot.scatter(range(0, 404), list)
    matplotlib.pyplot.title(i)

matplotlib.pyplot.show()


