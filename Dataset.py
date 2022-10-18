import torch
import numpy
import os
import torchaudio
import random
import copy
import csv

SAMPLE_RATE=12000
NUMBER_OF_SUMPLES=12000
CHECK_POINT=0.016
path_of_the_directory ="D:\Python\PythonProjects\AudioAI\\archive\\free-spoken-digit-dataset-master\\recordings"
digits_mapping = []
Recordings=3000 #Folder recordings
digits={"0":[],"1":[],
        "2":[],"3":[],
        "4":[],"5":[],
        "6":[],"7":[],
        "8":[],"9":[]}

#Augmentations
frequencyMasking=torchaudio.transforms.FrequencyMasking(freq_mask_param=35)#Put "white" lines in frequency domain
timeMasking=torchaudio.transforms.TimeMasking(time_mask_param=35)


#Create mel spectogram (kernel)
melSpectogram=torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=512,
        hop_length=256,
        n_mels=32
)

#Delete or Insert samples in signal if it needed
def change_signal_samples(signal):
    length_of_signal=signal.shape[1]
    if length_of_signal>NUMBER_OF_SUMPLES:
        signal=signal[:,:NUMBER_OF_SUMPLES]

    elif length_of_signal<NUMBER_OF_SUMPLES:
        missing_sumples=NUMBER_OF_SUMPLES-length_of_signal
        last_dim_padding = (0,missing_sumples)
        signal = torch.nn.functional.pad(signal,last_dim_padding)

    return signal

#Resample signal in 12000 sample rate
def resample_if_needed(signal,sr):
    if sr!=SAMPLE_RATE:
        resample=torchaudio.transforms.Resample(sr,SAMPLE_RATE)
        signal=resample(signal)
    return signal

def signal_conv(signal):
    signal = signal[0].numpy()  # convert to numpy
    signal = signal.flatten()  # flatten the signal
    signal = signal.tolist()
    return signal

#Manage dataset
def CreateDataset(mel_spectogram):

    object = os.scandir(path_of_the_directory)
    for n in object:
        if n.is_dir() or n.is_file():
            signal,sr = torchaudio.load(n)
            signal=resample_if_needed(signal,sr)
            signal=change_signal_samples(signal)
            signal=mel_spectogram(signal)

            if n.name.startswith('0'):
                digits["0"].append(signal)
                signal=signal_conv(signal)
                signal.insert(0,0)
                digits_mapping.append(signal)

            elif n.name.startswith('1'):
                digits["1"].append(signal)
                signal=signal_conv(signal)
                signal.insert(0,1)
                digits_mapping.append(signal)

            elif n.name.startswith('2'):
                digits["2"].append(signal)
                signal = signal_conv(signal)
                signal.insert(0,2)
                digits_mapping.append(signal)

            elif n.name.startswith('3'):
                digits["3"].append(signal)
                signal = signal_conv(signal)
                signal.insert(0,3)
                digits_mapping.append(signal)

            elif n.name.startswith('4'):
                digits["4"].append(signal)
                signal = signal_conv(signal)
                signal.insert(0,4)
                digits_mapping.append(signal)

            elif n.name.startswith('5'):
                digits["5"].append(signal)
                signal = signal_conv(signal)
                signal.insert(0,5)
                digits_mapping.append(signal)

            elif n.name.startswith('6'):
                digits["6"].append(signal)
                signal = signal_conv(signal)
                signal.insert(0,6)
                digits_mapping.append(signal)

            elif n.name.startswith('7'):
                digits["7"].append(signal)
                signal = signal_conv(signal)
                signal.insert(0,7)
                digits_mapping.append(signal)

            elif n.name.startswith('8'):
                digits["8"].append(signal)
                signal = signal_conv(signal)
                signal.insert(0,8)
                digits_mapping.append(signal)

            elif n.name.startswith('9'):
                digits["9"].append(signal)
                signal = signal_conv(signal)
                signal.insert(0,9)
                digits_mapping.append(signal)
    object.close()

# def train_data_split(factor):
#
#     random.shuffle(digits_mapping)
#     signal_samples=len(digits_mapping)
#     number_of_tested_samples=int(round(factor*signal_samples))
#     train_data=copy.deepcopy(digits_mapping[number_of_tested_samples:signal_samples])
#     test_data=copy.deepcopy(digits_mapping[:number_of_tested_samples])
#
#     return train_data,test_data

#Choose 5 random samples per digit and augment them
def augmentention():
    for key,value in digits.items():
        sample_list = random.sample(value, 5)
        for sample in sample_list:
            fr_sample=frequencyMasking(sample)
            time_sample=timeMasking(sample)

            fr_sample=signal_conv(fr_sample)
            time_sample=signal_conv(time_sample)

            time_sample.insert(0,int(key))
            fr_sample.insert(0,int(key))

            digits_mapping.append(time_sample)
            digits_mapping.append(fr_sample)

def main():
    print("Now dataset will be created")
    CreateDataset(melSpectogram)
    augmentention()
    with open('dataset.csv', 'w') as f:
        for sublist in digits_mapping:
            i = 0
            for item in sublist:
                i+=1
                if(i==1505):
                    f.write(str(item))
                else:
                    f.write(str(item) + ',')
            f.write('\n')
    print('Dataset created successfully and stored at dataset.csv file')
