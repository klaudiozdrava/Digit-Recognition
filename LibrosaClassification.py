import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

PREFIX='Foreground'

def ForeGroundClassifier(WAVNAME,PATHNAME):

    FILE_PATH=PATHNAME+'\\'+PREFIX+WAVNAME

    signal, sr = librosa.load(WAVNAME)
    plt.plot(signal, color='green')
    #Υπολογιζουμε το μεγεθος του σηματος μεσω του stft
    Signal_magnitude, phase = librosa.magphase(librosa.stft(signal))

    #Υπολογιζουμε το φιλτρο για τον background ηχο
    S_filter = librosa.decompose.nn_filter(Signal_magnitude,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))


    S_filter = np.minimum(Signal_magnitude, S_filter)

    margin_i, margin_v = 2, 10
    power = 2

    #οι μασκες των background και foreground στιγμιοτυπων του σηματος
    #Εφαρμοζεται η συναρτηση softmax για λογους βελτίωσης των αποτελεσμάτων
    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (Signal_magnitude - S_filter),
                                   power=power)

    mask_v = librosa.util.softmask(Signal_magnitude - S_filter,
                                   margin_v * S_filter,
                                   power=power)


    S_foreground = mask_v * Signal_magnitude
    S_background = mask_i * Signal_magnitude


    S_foreground=S_foreground*phase # Πολλαπλασιάζουμε τον foreground ηχο με την αρχικη φαση του σηματος
    S_ifft=librosa.istft(S_foreground) #Εφαρμοζεται ο inverse stft
    plt.plot(S_ifft,color='red')
    plt.show()

    sf.write(FILE_PATH,S_ifft,sr)

    return FILE_PATH



