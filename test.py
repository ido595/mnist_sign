import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import pathlib
import os
import random
import sklearn.preprocessing as pr

# disable gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TF to use only the CPU

# params
IMG_HEIGHT = 28
IMG_WIDTH = 28
SOFT_MAX = 25
EPOCH = 10
class_names = ["A", "B", "C", "D", "E", "F",
               "G", "H", "I", "K", "L", "M",
               "N", "O", "P", "Q", "R", "S",
               "T", "U", "V", "W", "X", "Y", "Z"]


def basic_CNN_model():
    _model = tf.keras.models.Sequential(name="basic_CNN")  # creating a sequential model for our CNN
    _model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
    _model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    _model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    _model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    _model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

    _model.add(tf.keras.layers.Flatten())  # flatten the output into vector
    _model.add(tf.keras.layers.Dense(64, activation='relu'))
    _model.add(tf.keras.layers.Dense(SOFT_MAX, activation='softmax'))  # 7 output layers for the features
    # softmax is better for single label prediction, sigmoid is the way to go with multi-label prediction
    _model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    _model.summary()
    return _model


def basic_model():
    return tf.keras.models.Sequential(name="basic_model", layers=[
        tf.keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(SOFT_MAX, activation='softmax')
    ])


def smaller_VGGNET_model():
    _model = tf.keras.models.Sequential(name="smaller_VGGNET")
    # CONV => RELU => POOL
    _model.add(  # padding = "same" results in padding the input such that the output has the same length
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
    _model.add(tf.keras.layers.BatchNormalization(axis=-1))
    _model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    _model.add(tf.keras.layers.Dropout(0.25))

    # (CONV => RELU) * 2 => POOL
    _model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
    _model.add(tf.keras.layers.BatchNormalization(axis=-1))
    _model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
    _model.add(tf.keras.layers.BatchNormalization(axis=-1))
    _model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    _model.add(tf.keras.layers.Dropout(0.25))

    # (CONV => RELU) * 2 => POOL
    _model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same"))
    _model.add(tf.keras.layers.BatchNormalization(axis=-1))
    _model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same"))
    _model.add(tf.keras.layers.BatchNormalization(axis=-1))
    _model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    _model.add(tf.keras.layers.Dropout(0.25))

    # first (and only) set of FC => RELU layers
    _model.add(tf.keras.layers.Flatten())
    _model.add(tf.keras.layers.Dense(1024, activation='relu'))
    _model.add(tf.keras.layers.BatchNormalization())
    _model.add(tf.keras.layers.Dropout(0.5))

    # softmax classifier
    _model.add(tf.keras.layers.Dense(SOFT_MAX, activation="softmax"))  # 7 features
    _model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    _model.summary()
    # return the constructed network architecture
    return _model

# Read training and test data files
train = pd.read_csv("sign_mnist_train/sign_mnist_train.csv").values
test = pd.read_csv("sign_mnist_test/sign_mnist_test.csv").sample(frac=1).reset_index(drop=True).values

# Reshape and normalize training data
x_train = train[:, 1:].reshape(train.shape[0], 28, 28, 1).astype('float32')
x_train = x_train / 255.0
y_train = train[:, 0]

# Reshape and normalize test data
x_test = test[:, 1:].reshape(test.shape[0], 28, 28, 1).astype('float32')
x_test = x_test / 255.0
y_test = test[:, 0]

#model = basic_CNN_model()
model = basic_model()
# model =smallerVGGNET_model()

tf.keras.utils.plot_model(
    model,
    to_file='res_' + model.name + "_architecture.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=True,
    dpi=96
)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Create a callback that saves the model's weights
checkpoint_path = "res_" + str(EPOCH) + "_" + model.name + "_data/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# write text file
model.save_weights(checkpoint_path.format(epoch=0))

train_history = model.fit(x_train, y_train,
                          callbacks=[cp_callback],
                          validation_data=(x_test, y_test),
                          epochs=EPOCH, verbose=2)

# save simple data to text file
# convert the history.history dict to a pandas DataFrame:
hist_df = pd.DataFrame(train_history.history)

# or save to csv:
hist_csv_file = "res_" + model.name + "_" + str(EPOCH) +".csv"
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# plotting
print("PLOTTING")

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img.reshape(IMG_WIDTH, IMG_HEIGHT), cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(SOFT_MAX), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# plotting the images
predictions = model.predict(x_test)
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], y_test, x_test)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], y_test)
plt.tight_layout()
plt.savefig("res_" + model.name + "_" + str(EPOCH)  + "_images.png")
plt.show()

# plotting the accuracy
plt.plot(train_history.history["accuracy"], label="Train")
plt.plot(train_history.history["val_accuracy"], label="Test")
plt.title("Training&Test Accuracy " + model.name)
plt.xlabel("Epoch " + str(EPOCH))
plt.ylabel("Accuracy")
plt.legend(loc="upper left")
plt.savefig("res_" + model.name + "_" + str(EPOCH) + "_graph.png")
plt.show()
