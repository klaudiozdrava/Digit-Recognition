import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


DIGIT_FRAME_SIZE=12000

#Υπολογιζει την αρχη και το τελος των ψηφιων του σηματος
#Η ιδεα ειναι να αποκτησουμε οσο το δυνατον καλυτερο ευρος σηματος για το καθε ψηφιο
#θεωρουμε αρχικα πως το peak του φηφιου ειναι στο κεντρο του διαστηματος του σηματος του συγκεκριμνεου ψηφιου
#Επειτα με επαναληψεις μετακινουμε το κεντρο αυτο αριστερα η δεξια ωστε να αποκτησουμε το καταλληλο ευρος
def extractDigits(peaks,frame_size,scalered_signal):

    start_end_of_digits=[]
    cc=[]
    for peak in peaks:
        leftside = int(frame_size / 2)
        rightsize = int(frame_size / 2 + 1)
        center = peak#position of peak.We suugest that is in the center of digit signal
        if (center - leftside < 0):
            start_end_of_digits.append((0, frame_size))
            continue
        elif (center+rightsize>len(scalered_signal)):
            start_end_of_digits.append((len(scalered_signal)-frame_size, len(scalered_signal)-1))
            continue

        sum_of_leftside = sum(scalered_signal[center - leftside : center])
        sum_of_rightside = sum(scalered_signal[center : center + rightsize])
        right_right_side=sum(scalered_signal[center + rightsize : center + rightsize + abs(center-leftside)])
        left_left_side=sum(scalered_signal[center-leftside-abs(center+rightsize):center-leftside])
        if (sum_of_rightside > sum_of_leftside):
            while ((sum_of_rightside > sum_of_leftside) and (center<peak+frame_size))  :
                center += 50
                sum_of_leftside = sum(scalered_signal[center - leftside : center])
                sum_of_rightside = sum(scalered_signal[center : center + rightsize])
            start_end_of_digits.append((center - leftside -100, center + rightsize -100))
        else:
            while ((sum_of_rightside < sum_of_leftside) and (center>peak-frame_size)):
                center -= 50
                sum_of_leftside = sum(scalered_signal[center - leftside:center])
                sum_of_rightside = sum(scalered_signal[center:center + rightsize])
            start_end_of_digits.append((center - leftside +80 , center + rightsize + 80 ))

    return start_end_of_digits

#geometric mean
def g_mean(x):
    a = np.log(x)
    return np.exp(a.mean())

#Αφαιρουμε στοιχεια απο την λιστα που δεν αντιπροσωπευουν αριθμους με χρηση του γεωμετρικου μεσου
def remove_unecessary_peaks(original_signal,signal,digits_offsets):
    only_digits=[]
    plt.subplot(1, 2, 2)
    plt.plot(original_signal)
    plt.title("Original Signal")
    plt.ylabel("Amplitude")
    plt.xlabel("Samples")
    results = [sum(signal[offset[0]:offset[1]]) for offset in digits_offsets]
    mean=g_mean(results)#Get the geometric mean of areas of all digits and some little noises

    for pos,area in enumerate(results):#Ελεγχει εαν το εμβαδον της καθε περιοχης ειναι μεγαλυτερο απο το γεωμετρικο μεσο ολων των περιοχων
        if area>mean:#Εαν ισχυει τοτε σημαινει πως υπαρχει ψηφιο οποτε και αποθηκευουμε την αρχη και τελος του ψηφιου αυτου πανω στο
            # αρχικο σημα
            start_offset = digits_offsets[pos][0]
            end_offset = digits_offsets[pos][1]
            only_digits.append(original_signal[start_offset:end_offset])
            x_axis=[i for i in range(start_offset,end_offset)]
            maximum=max(original_signal[start_offset:end_offset])
            minimum=min(original_signal[start_offset:end_offset])
            minY = [minimum for j in range(start_offset, end_offset)]
            maxY = [maximum for i in range(start_offset, end_offset)]
            plt.fill_between(x_axis, minY, color='green', alpha=.3)
            plt.fill_between(x_axis,maxY,color='green', alpha=.3)
    plt.show()
    return only_digits

#Get the signal from user
#Convert it to have only positive numbers
#Find where digits appeared
def analyzeSignal(sound):

    signal,sr =librosa.load(sound)
    signal_non_negative = abs(signal)#Convert all values to positive
    smaller = np.min(signal_non_negative[np.nonzero(signal_non_negative)])#Get smaller non negative value
    scaler = np.rint(1 / smaller)#
    signal_non_negative = signal_non_negative * scaler #Scale the original signal
    peaks, _ = find_peaks(signal_non_negative, distance=2*DIGIT_FRAME_SIZE)#Get peaks of signal
    plt.subplot(1, 2, 1)
    plt.plot(signal_non_negative)
    plt.plot(peaks, signal_non_negative[peaks], "x")
    plt.title("Scalered Signal")
    plt.ylabel('Scalered Amplitude ')
    plt.xlabel('Samples')
    digits_offset=extractDigits(peaks,DIGIT_FRAME_SIZE,signal_non_negative)
    only_digits=remove_unecessary_peaks(signal,signal_non_negative,digits_offset)

    return only_digits,sr

