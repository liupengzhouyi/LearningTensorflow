import numpy
import tensorflow
import tensorflow_hub
import tensorflow_datasets

train_validation_split = tensorflow_datasets.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tensorflow_datasets.load(
    name='imdb_reviews',
    split=(train_validation_split, tensorflow_datasets.Split.TEST),
    as_supervised=True)


train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

# print(train_examples_batch)
# print(train_labels_batch)

#  二分类

embedding = 'https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1'
hub_layer = tensorflow_hub.KerasLayer(
    embedding,
    input_shape=[],
    dtype=tensorflow.string,
    trainable=True)

# print(hub_layer(train_examples_batch[:3]))


model = tensorflow.keras.Sequential()
model.add(hub_layer)
model.add(tensorflow.keras.layers.Dense(units=16, activation='relu'))
model.add(tensorflow.keras.layers.Dense(units=1, activation='sigmoid'))

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=5,
                    validation_data = validation_data.batch(512),
                    verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, validation_data))