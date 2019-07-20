
from pathlib import Path
import librosa
from pprint import pprint
import pickle
import matplotlib.pyplot as plt
import librosa.display
import numpy as np


def getAudioDuration():
    audioDuration = dict()
    dataSetPath = Path(".\\audio")
    for childDir in dataSetPath.iterdir():
        for audioFiles in childDir.iterdir():
                audioDuration[audioFiles] = librosa.get_duration(filename=str(audioFiles))
    return audioDuration;




def getAudioParts():
    audioParts = dict()
    for audioFile, duration in getAudioDuration().items():
        audioParts[audioFile] = duration//30
    return audioParts



def getAudioSamples():
    audioSamples = dict()
    for audioFile, duration in getAudioParts().items():
        librosaSamples = []
        for part in range(int(duration)):
            librosaSamples.append(librosa.load(str(audioFile), offset=30*part, duration=30))
        audioSamples[audioFile] = librosaSamples
    return audioSamples

def saveAudioSamples():
    with open ("librosaDeatils.pkl", "wb", ) as f:
        pickle.dump(getAudioSamples(),f)

def loadAudioSamples():
    with open(".\\librosaDeatils.pkl", "rb") as f:
        return pickle.load(f)

def getChroma(y, sr):
    return librosa.feature.chroma_cqt(y=y, sr=sr)

def getYHarmonic(y):
    return librosa.effects.harmonic(y=y, margin=8)

def buildDatatset():
    audioSamples = loadAudioSamples()
    dataset = []
    for audioFile in audioSamples:
        for y,sr in audioSamples[audioFile]:
            chroma = getChroma(getYHarmonic(y), sr)
            chroma_mean = np.mean(chroma, axis = 1)
            chroma_std = np.std(chroma, axis = 1)
            chroma_median = np.median(chroma, axis =1)
            features = []
            features.extend(chroma_mean)
            features.extend(chroma_std)
            features.extend(chroma_median)
            target = audioFile.parts[-2]
            row = {target: features}
            dataset.append(row)
    return dataset

def saveDataset(dataset):
    with open(".\\dataset.pkl", "wb") as datasetFileObj:
        pickle.dump(dataset, datasetFileObj)

def loadDataset():
    with open(".\\dataset.pkl", "rb") as datasetFileObj:
        return pickle.load(datasetFileObj)

if __name__ == "__main__":
    datasetLoaded = loadDataset()
    pass
