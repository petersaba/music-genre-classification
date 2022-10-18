import os

SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048
NB_SEGMENTS = 5
JSON_FILENAME = 'dataset.json'

def save_mfccs(dataset_dir_path, json_path=JSON_FILENAME, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_fft=N_FFT, nb_segments=NB_SEGMENTS):

    for i, (root, dir, filename) in enumerate(os.walk(dataset_dir_path)):
        print(root)

if __name__ == '__main__':
    os.chdir('dataset')
    save_mfccs('genres_original')