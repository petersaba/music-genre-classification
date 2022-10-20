import json
from pickletools import optimize
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.python.keras as keras

DATASET_PATH = './dataset/dataset.json'

def load_dataset(dataset_path):

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    inputs = np.array(data['mfcc'])
    targets = np.array(data['labels'])

    return inputs, targets

if __name__ == '__main__':

    inputs, targets = load_dataset(DATASET_PATH)
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3)

    # Sequential model: moves from the left to the right 
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        # relu is generally better fir training models than sigmoid and most importantly
        # because it avoids the vanishing gradient problem as derivative of sigmoid func is 0.25 at most
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(64, activation='relu'),

        # nb of neurons is the same as the number of options in this case 10 genres
        keras.layers.Dense(10, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test), epochs=50, batch_size=32)