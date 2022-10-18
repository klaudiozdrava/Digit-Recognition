import torch
import numpy
import torchaudio
from Dataset import resample_if_needed
from Network import FeedForwordNetwork
import matplotlib.pyplot as plt
import librosa
from torch import nn
from ExtractDigits import analyzeSignal
from LibrosaClassification import ForeGroundClassifier
import os

SAMPLE_RATE=12000
NUMBER_OF_SUMPLES=12000
INPUT_SIZE=1504 #Dataset row length
FIRST_LAYER=512
SECOND_LAYER=64
THIRD_LAYER=64
NUM_CLASSES=10
SOUND_NAME='testingForG.wav'
PREFIX='Foreground'

class_mapping=[
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

melSpectogram=torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=512,
        hop_length=256,
        n_mels=32
)

def change_signal_samples(signal):
    length_of_signal=len(signal)

    if length_of_signal>NUMBER_OF_SUMPLES:
        signal=signal[:,:NUMBER_OF_SUMPLES]

    elif length_of_signal<NUMBER_OF_SUMPLES:
        missing_sumples=NUMBER_OF_SUMPLES-length_of_signal
        last_dim_padding = (0,missing_sumples)
        signal = torch.nn.functional.pad(signal,last_dim_padding)

    return signal

def predict(model,input,class_mapping):
    model.eval()
    normalization = nn.Softmax(dim=0)
    with torch.no_grad():
        prediction=model(input)
        prediction=normalization(prediction)
        predictide_index= torch.argmax(prediction)
        predicted = class_mapping[predictide_index]
    return predicted

def MakePrediction(signal_input,sr):

    signal=torch.from_numpy(signal_input)
    signal = resample_if_needed(signal, sr)
    signal = change_signal_samples(signal)
    signal = melSpectogram(signal)
    signal=signal.numpy()
    signal=signal.flatten()
    signal=torch.from_numpy(signal)

    return signal


if __name__ == '__main__':

    expected_digits=[0,9,8,0,2,6]
    valid_predictions=[]
    SOUND_NAME=input('Insert path of audio file (must be in waveform) : ')

    if os.path.isfile(SOUND_NAME):
        SUFFIX=os.path.basename(SOUND_NAME)
        PATH=os.path.dirname(SOUND_NAME)
        FOREGOUNDSOUND=ForeGroundClassifier(SUFFIX,PATH)
        if os.path.isfile(FOREGOUNDSOUND):
            digits,sample_rate=analyzeSignal(FOREGOUNDSOUND)
            net =FeedForwordNetwork(INPUT_SIZE,FIRST_LAYER,SECOND_LAYER,NUM_CLASSES)
            state_dict=torch.load('feedforward.pth')
            net.load_state_dict(state_dict)
            print(f'The prediction of input signal {FOREGOUNDSOUND} is:')
            for digit in digits:
                data=MakePrediction(digit,sample_rate)
                predicted=predict(net,data,class_mapping)
                print(f"Predicted digit : {predicted} |")
