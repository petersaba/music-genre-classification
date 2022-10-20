import os
import librosa
import json

SAMPLE_RATE = 22050
SAMPLES_PER_TRACK = 22050 * 30
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

    samples_per_segment = int(SAMPLES_PER_TRACK / nb_segments)

    for i, (root, dir_array, file_array) in enumerate(os.walk(dataset_dir_path)):
        
        if root != dataset_dir_path:
            i -= 1 # first value of i is for the root directory which cannot be used to label the audio genres
            for file in file_array:
                    signal, sr = librosa.load(f'./{root}/{file}', sr=SAMPLE_RATE)

                    for current_segment in range(nb_segments):
                        segment_start = samples_per_segment * current_segment
                        segment_end = samples_per_segment * (current_segment + 1)

                        segment_samples = signal[segment_start:segment_end]
                        mfcc = librosa.feature.mfcc(y=segment_samples, hop_length=hop_length, n_mfcc=13, n_fft=n_fft)
                        mfcc = mfcc.T
                        data['mfcc'].append(mfcc.tolist())
                        data['labels'].append(i)
                    print(f'{file} has been read')
        else:
            for dir in dir_array:
                data['genres'].append(dir)

    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)
        

if __name__ == '__main__':
    os.chdir('dataset')
    save_mfccs('genres_original')