import json
import numpy as np

DATASET_PATH = './dataset/dataset.json'

def load_dataset(dataset_path):

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    inputs = np.array(data['mfcc'])
    targets = np.array(data['labels'])

    return inputs, targets

if __name__ == '__main__':

    inputs, targets = load_dataset(DATASET_PATH)