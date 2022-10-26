from json import load
from genre_classifier import DATASET_PATH, load_dataset
# from genre_classifier import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np

def prepare_data(test_size, validation_size, dataset_path=DATASET_PATH):

    X, y = load_dataset(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=validation_size)

    # CNNs inputs always must have an extra dimension(generally for rgb) but audio has one dimension
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validate = X_validate[..., np.newaxis]

    return X_train, X_test, X_validate, y_train, y_test, y_validate

if __name__ == "__main__":
    X_train, X_test, X_validate, y_train, y_test, y_validate = prepare_data(0.25, 0.2)