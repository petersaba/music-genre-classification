import os
import librosa
import json

SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048
NB_SEGMENTS = 10 # each audio will be devided into 10 parts in order to increase the data in the dataset
JSON_FILENAME = 'dataset.json'

def save_mfccs(dataset_dir_path, json_path=JSON_FILENAME, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_fft=N_FFT, nb_segments=NB_SEGMENTS):

    data = {
        "genres": [],
        "mfcc": [],
        "labels": []
    }

    for i, (root, dir_array, file_array) in enumerate(os.walk(dataset_dir_path)):
        
        if root != dataset_dir_path:
            i -= 1 # first value of i is for the root directory which cannot be used to label the audio genres
            for file in file_array:
                signal, sr = librosa.load(f'./{root}/{file}', sr=SAMPLE_RATE)
                mfcc = librosa.feature.mfcc(y=signal, hop_length=hop_length, n_mfcc=13, n_fft=n_fft)
                data['mfcc'].append(mfcc.tolist())
                data['labels'].append(i)
                print(f'{file} has been read')
            pass
        else:
            for dir in dir_array:
                data['genres'].append(dir)

    # print(data['labels'])
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)
        

if __name__ == '__main__':
    os.chdir('dataset')
    save_mfccs('genres_original')