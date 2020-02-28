import os

import tensorflow

(train_images, train_labels), (test_images, test_labels) = tensorflow.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# 定义一个简单的序列模型
def create_model():
    model = tensorflow.keras.Sequential()
    model.add(tensorflow.keras.layers.Dense(units=512, activation='relu', input_shape=(784, ))),
    model.add(tensorflow.keras.layers.Dropout(0.2))
    model.add(tensorflow.keras.layers.Dense(units=10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
#
# # 创建一个基本的模型实例
# model = create_model()
# #
# # 显示模型的结构
# model.summary()
#
# checkpoint_path = "training_1/cp.ckpt"
#
# # 返回路径 ： dirname
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# print("----------", checkpoint_dir)

#
# # 创建一个保存模型权重的回调
# cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     save_weights_only=True,
#     verbose=1,
# )
#
# # 使用新的回调训练模型
# model.fit(train_images,
#           train_labels,
#           epochs=10,
#           validation_data=(test_images, test_labels),
#           callbacks=[cp_callback], # 通过回调训练
#           )
#
# # 这可能会生成与保存优化程序状态相关的警告。
# # 这些警告（以及整个笔记本中的类似警告）是防止过时使用，可以忽略。
#
#
# # 创建一个基本模型实例
# model = create_model()
#
# # 评估模型
# loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
#
# print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
#

# 从checkpoint加载权重并重新评估
# 加载权重
# model.load_weights(checkpoint_path)
#
# # 重新评估模型
# loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
#

# 在文件名中包含 epoch (使用 `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# # 创建一个回调，每 5 个 epochs 保存模型的权重
# cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     verbose=1,
#     save_weights_only=True,
#     period=5)
#
# # 创建一个新的模型实例
# model = create_model()
#
# # 使用 `checkpoint_path` 格式保存权重
# model.save_weights(checkpoint_path.format(epoch=0))
#
# # 使用新的回调*训练*模型
# model.fit(train_images,
#               train_labels,
#               epochs=50,
#               callbacks=[cp_callback],
#               validation_data=(test_images,test_labels),
#               verbose=0)
#


# latest = tensorflow.train.latest_checkpoint(checkpoint_dir)
#
# # 创建一个新的模型实例
# model = create_model()
#
# # 加载以前保存的权重
# model.load_weights(latest)
#
# # 重新评估模型
# loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# 保存权重
# model.save_weights('./checkpoints/my_checkpoint')

# 创建模型实例
# model = create_model()
#
# # 加载权重
# model.load_weights('./checkpoints/my_checkpoint')
#
# # Evaluate the model
# loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))



# # 创建一个新的模型实例
# model = create_model()
#
# # 训练模型
# model.fit(train_images, train_labels, epochs=5)
#
# # 将整个模型保存为HDF5文件
# model.save('my_model.h5')


# 重新创建完全相同的模型，包括其权重和优化程序
new_model = tensorflow.keras.models.load_model('my_model.h5')

# 显示网络结构
new_model.summary()


loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

