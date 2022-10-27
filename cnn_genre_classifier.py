from json import load
from genre_classifier import DATASET_PATH, load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

def prepare_data(test_size, validation_size, dataset_path=DATASET_PATH):

    X, y = load_dataset(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=validation_size)

    # CNNs inputs always must have an extra dimension(generally for rgb) but audio has one dimension
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validate = X_validate[..., np.newaxis]

    return X_train, X_test, X_validate, y_train, y_test, y_validate

def create_model(input_shape):

    keras = tf.keras
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

if __name__ == "__main__":
    X_train, X_test, X_validate, y_train, y_test, y_validate = prepare_data(0.25, 0.2)

    # print(type(X_train))
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = create_model(input_shape)