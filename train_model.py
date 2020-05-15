import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

in_pickle = open('dataset.pickle', 'rb')
dataset = pickle.load(in_pickle)

# train_len = np.around(len(dataset) * (80/100))
# train_len = int(train_len)
# train_data = dataset[:train_len]
# test_data = dataset[train_len:]

x_train = []
y_train = []

for x, y in dataset[:300]:
    x_train.append(x)
    y_train.append(y)
print(len(x_train))
x_test = []
y_test = []

for x, y in dataset[300:]:
    x_test.append(x)
    y_test.append(y)

print(len(x_test))
x_train = np.array(x_train)
x_test = np.array(x_test)

x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0], 50, 50, 1)
x_test = x_test.reshape(x_test.shape[0], 50, 50, 1)
print(x_train.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, input_shape=(50, 50, 1), activation='relu', kernel_size=(2, 2)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, activation='relu', kernel_size=(2, 2)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=15, validation_split=0.2)
print('eval')
his = model.evaluate(x_test, y_test)
model.save_weights('model.h5', overwrite=True)
print(his)
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
