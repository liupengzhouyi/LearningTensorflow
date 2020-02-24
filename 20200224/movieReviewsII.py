import tensorflow
import numpy
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

imdb = tensorflow.keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()

word_index = {k:(v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

train_data = tensorflow.keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)

test_data = tensorflow.keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)

model = tensorflow.keras.Sequential()
model.add(tensorflow.keras.layers.Embedding(input_dim=10000, output_dim=16))
model.add(tensorflow.keras.layers.GlobalAveragePooling1D())
model.add(tensorflow.keras.layers.Dense(units=16, activation='relu'))
model.add(tensorflow.keras.layers.Dense(units=1, activation='sigmoid'))

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    train_data[10000:],
    train_labels[10000:],
    epochs=13,
    batch_size=512,
    validation_data=(
        train_data[:10000],
        train_labels[:10000]),
    verbose=1)


# results = model.evaluate(test_data, test_labels, verbose=2)
# print(results)


history_dict = history.history
print(history_dict)

acc = history_dict['accuracy']
acc_value = history_dict['val_accuracy']
loss = history_dict['loss']
loss_value = history_dict['val_loss']

lengths = range(1, len(acc) + 1)

# plt.plot(lengths, loss, label='training loss')
# plt.plot(lengths, loss_value, label='validation loss')
plt.plot(lengths, acc, label='training acc')
plt.plot(lengths, acc_value, label='validation acc')
plt.legend()
plt.savefig('image001.png')
plt.show()





