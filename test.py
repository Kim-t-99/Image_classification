import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# THERE IS NOT ALOT OF COMMENTS ON THIS PYTHON FILE
# This file is not apart of the classification properly and only a side.
#
# This file is used purley to test number of epochs.
# This is because pycharm and/or my pc could not handle it in the param grid
#
#
#


data_dir = "data"

def load_image(path):
    images = []
    labels = []

    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            img = img / 255.0

            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

def make_model():
    model = Sequential()

    model.add(Input(shape=(64, 64, 1)))

    # First Convolution Layer
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Second Convolution Layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Third Convolution Layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    # Dense Layers
    model.add(Dense(128, activation='relu'))

    model.add(Dense(4, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

images, labels = load_image(data_dir)


# Label encoding
label_enc = LabelEncoder()
integer_labels = label_enc.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(images, integer_labels, test_size=0.25, random_state=42)

# Reshape for CNN input
X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
X_test = X_test.reshape(X_test.shape[0], 64, 64, 1)

# One-Hot Encoding
y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=4)

epochs_test = [5, 7, 10, 12]

best_accuracy = 0
best_model = None
best_epochs = None

for epochs in epochs_test:

    print(f'Training model with {epochs} epochs')

    model = make_model()
    model.fit(X_train, y_train_oh, epochs=epochs, batch_size=32, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test_oh)
    print(f'Accuracy for {epochs} epochs: {accuracy}')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model, best_epochs = model, epochs

# Train the model
print(f'Best number of epochs: {best_epochs} ')
print(f'Test accuracy: {accuracy}')
print(best_model.summary())

