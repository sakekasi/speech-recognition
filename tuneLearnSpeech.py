
# coding: utf-8
import warnings
warnings.filterwarnings('ignore')

from os import listdir
from os.path import isfile, join
import pickle

from hmmlearn import hmm

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plot
import numpy

from sklearn.model_selection import train_test_split
from skopt import gp_minimize

ghmms = {}
wordData = {}
data = {}
datadirs = [f for f in listdir("data/") if ('.' not in f)]
for directory in datadirs:
        curdir = "data/" + directory
        datafiles = [f for f in listdir(curdir) if isfile(join(curdir,f))]

        dataForWord = []
        for f in datafiles:
            (rate, signal) = wav.read(curdir + "/" + f)
            dataForWord.append(mfcc(signal, rate, winfunc=numpy.hamming))

        wordData[directory] = dataForWord
        
        train, test = train_test_split(dataForWord, test_size=0.3)
        data[directory] = {'train': train, 'test': test}

def predict(fileData):
    logOddsToKey = {}
    for key in ghmms:
        ghmm = ghmms[key]
        logOdds = ghmms[key].score_samples(fileData)[0]
        logOddsToKey[logOdds] = key
    return logOddsToKey[max(logOddsToKey.keys())]

def objective(params):
    n_components, n_iter = params

    for directory in datadirs:
        train, test = train_test_split(wordData[directory], test_size=0.3)
        data[directory] = {'train': train, 'test': test}

    flatData = {}
    lengths = {}
    for directory in datadirs:
        flatDataForWord = []
        runLengthsForWord = []
        for fileData in data[directory]['train']:
            runLengthsForWord.append(len(fileData))
            flatDataForWord += fileData.flatten().tolist()
                
        flatData[directory] = numpy.array(flatDataForWord).reshape(-1, 13)
        lengths[directory] = runLengthsForWord

    for directory in datadirs:
        ghmm = hmm.GaussianHMM(n_components=n_components,n_iter=n_iter)
        ghmm.fit(flatData[directory], lengths=lengths[directory])
        ghmms[directory] = ghmm

    labeledTestData = []
    for key in data:
        for fileData in data[key]['test']:
            labeledTestData.append((fileData, key))

    successful = 0
    for test in labeledTestData:
        prediction = predict(test[0])
        if prediction == test[1]:
            successful += 1

    return ( 1.0 - float(successful) / len(labeledTestData) )

space = [
            (1, 50), # n_components
            (1, 25)  # n_iter
        ]

res_gp = gp_minimize(objective, space, n_calls=100, random_state=0)
print("Best score=%.4f" % res_gp.fun)
print("""Best parameters:
- n_components=%d
- n_iter=%d""" % (res_gp.x[0], res_gp.x[1]))



