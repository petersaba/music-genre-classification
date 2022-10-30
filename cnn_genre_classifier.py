from json import load
from genre_classifier import DATASET_PATH, load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from utilities import plot_model_history


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

    model.add(keras.layers.Conv2D(64, (3, 3), input_shape=input_shape))
    model.add(keras.layers.BatchNormalization()) # normalizing input before the activation
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))

    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dropout(0.96))

    model.add(keras.layers.Dense(10, activation='softmax'))

    return model, keras

if __name__ == "__main__":
    X_train, X_test, X_validate, y_train, y_test, y_validate = prepare_data(0.25, 0.2)

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model, keras = create_model(input_shape)

    optimizer = keras.optimizers.Adam(learning_rate=0.00005)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, validation_data=(X_validate, y_validate), batch_size=512, epochs=650)

    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f'loss is: {test_error}\naccuracy is: {test_accuracy}')

    plot_model_history(history)