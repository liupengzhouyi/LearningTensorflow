import tensorflow
import pandas
import seaborn
import matplotlib.pyplot
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# dataset_path = tensorflow.keras.utils.get_file(
#     "auto-mpg.data",
#     "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
# )
# print(dataset_path)

column_names = ['MPG', 'Cylinders', 'Displacement',
                'Horsepower', 'Weight', 'Acceleration',
                'Model Year', 'Origin']

raw_dataset = pandas.read_csv('.keras/datasets/auto-mpg.data',
                              names=column_names,
                              na_values='?', comment='\t',
                              sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())

print(dataset.isna().sum())

dataset = dataset.dropna()

origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0

print(dataset.tail())

train_dataset = dataset.sample(frac=0.8, random_state=0)

test_dataset = dataset.drop(train_dataset.index)

seaborn.pairplot(
    train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']],
    diag_kind='kde',
)

train_stats = train_dataset.describe()

train_stats.pop('MPG')

train_stats = train_stats.transpose()

print(train_stats)

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


def norm(x):
    return (x-train_stats['mean']) / train_stats['std']



normed_train_data = norm(train_dataset)

normed_test_data = norm(test_dataset)

def build_model():
    model = tensorflow.keras.Sequential()
    model.add(tensorflow.keras.layers.Dense(
        units=64,
        activation='relu',
        input_shape=[len(train_dataset.keys())]
    ))
    model.add(tensorflow.keras.layers.Dense(
        units=64,
        activation='relu',
    ))
    model.add(tensorflow.keras.layers.Dense(units=1))

    optimizer = tensorflow.keras.optimizers.RMSprop(learning_rate=0.001)

    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae', 'mse']
    )

    return model

model = build_model()

model.summary()

print(model.predict(normed_train_data[:10]))

class PrintDot(tensorflow.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            if epoch != 0:
                print(' ')
        else:
            print('.', end=' ')


history = model.fit(normed_train_data,
                    train_labels,
                    epochs=1000,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[PrintDot()]
                    )

print('')
hist = pandas.DataFrame(history.history)

hist['epoch'] = history.epoch

print(hist.tail())

print('-----over-----')
