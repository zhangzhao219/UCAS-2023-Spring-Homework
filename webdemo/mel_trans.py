import librosa
import librosa.display
import matplotlib.pyplot as plt


def Waveform(file_name, saved_name):
    samples, sr = librosa.load(file_name, sr=16000)
    plt.figure()
    librosa.display.waveshow(samples, sr=sr)
    plt.axis('off')
    plt.savefig('./static/images/'+saved_name)