import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D, Input
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
import joblib

data_dir = "data"

#Loading the images and using labeling classes with folder names
#
def load_image(path):
    images = []
    labels = []

    #loop over each subfoler(label)
    for label in os.listdir(path):
        label_path = os.path.join(path, label)


        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            #Reading images as grayscale. Then resizing and normalizing
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            img = img / 255.0

            images.append(img)
            labels.append(label)


    return np.array(images), np.array(labels)


#Creating a function for CNN model that can easily change parameters and img imput
def make_model(conv_1_filters=32, conv_2_filters=64, conv_3_filters=128,  learning_rate=0.0001,  pool_size=2):

    # Initiate the Sequential model
    model = Sequential()

    # Input layer that spacifies the shape of the input images
    model.add(Input(shape=(64, 64, 1)))

    # First Convolution Layer
    # Default values filter=32, 3x3 kernel, relu activiation
    model.add(Conv2D(conv_1_filters, (3, 3), activation='relu'))
    # Pooling layer to reduce dimensions of the output volume and hopefully reduce overfitting
    model.add(MaxPooling2D((pool_size, pool_size)))

    # Second Convolution Layer
    # Default values filter=64, 3x3 kernel, relu activiation
    model.add(Conv2D(conv_2_filters, (3, 3), activation='relu'))
    # Pooling layer to reduce dimensions of the output volume and hopefully reduce overfitting
    model.add(MaxPooling2D((pool_size, pool_size)))

    #third Convolution Layer
    # Default values filter=128, 3x3 kernel, relu activiation
    model.add(Conv2D(conv_3_filters, (3, 3), activation='relu'))
    # Pooling layer to reduce dimensions of the output volume and hopefully reduce overfitting
    model.add(MaxPooling2D((pool_size, pool_size)))

    # Flatten to create single feature vector
    model.add(Flatten())

    # Dense layers.  1st with 128 units and 2nd with 4 units because of 4 classes
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    #Defining the optimiser
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    #compile the model with categorical crossentropgy for a multi class classification model.
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


#parameter grid to test different values in the different parameters
param_grid = {
    'model__conv_1_filters': [32, 64],
    'model__conv_2_filters': [64, 128],
    'model__conv_3_filters': [128],
    'model__learning_rate': [0.0001, 0.001],
    'batch_size': [32, 64],
    'epochs': [10],
    'model__pool_size': [2],
}


#Defining images and labels with the load img function
images, labels = load_image(data_dir)

#labelencoding the categories for easy of use
label_enc = LabelEncoder()
integer_labels = label_enc.fit_transform(labels)

#Splitting the data into test and train. Using 0.25 test split
X_train, X_test, y_train, y_test = train_test_split(images, integer_labels, test_size=0.25, random_state=42)

# Reshape for CNN input
X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
X_test = X_test.reshape(X_test.shape[0], 64, 64, 1)

#keras model wrapper to initiate the CNN model
model = KerasClassifier(model=make_model, verbose=1)

# One-Hot Encoding
y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=4)

#Grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_results = grid_search.fit(X_train, y_train_oh)

print(f'Best Accuracy: {grid_results.best_score_} with {grid_search.best_params_}')

best_model = grid_results.best_estimator_.model_
loss, accuracy = best_model.evaluate(X_test, y_test_oh)
print(f'Test accuracy on the best model: {accuracy}')
print(best_model.summary())

#saving best model for use
best_model.save("best_model.h5")

joblib.dump(label_enc, "label_encoder.pkl")