import json
import numpy as np
from sklearn.model_selection import train_test_split

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