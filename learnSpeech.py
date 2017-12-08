import pickle

from hmmlearn import hmm

from os import listdir
from os.path import isfile, join

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plot
import numpy

from sklearn.model_selection import train_test_split

WORDDATA='worddata.p'
# SPEECHMODELS='speechmodels.p'

wordData = {}
finalTestData = {}
learningData = {}
loadedWordData = False
splittedFinalTestData = False

# FILE PROCESSING

def loadWordData():
  global wordData
  if loadedWordData:
    return wordData
  elif deserializeWordData():
    return wordData
  else:
    datadirs = [f for f in listdir("data/") if ('.' not in f)]
    for directory in datadirs:
      curdir = "data/" + directory
      datafiles = [f for f in listdir(curdir) if isfile(join(curdir,f))]
      dataForWord = []
      for f in datafiles:
        (rate, signal) = wav.read(curdir + "/" + f)
        dataForWord.append(mfcc(signal, rate, winfunc=numpy.hamming))
      wordData[directory] = dataForWord
    with open(WORDDATA, 'wb') as wordDataFile:
      pickle.dump(wordData, wordDataFile)
    return wordData

def deserializeWordData():
  global wordData
  try:
    with open(WORDDATA, 'rb') as wordDataFile:
      wordData = pickle.load(wordDataFile)
      loadedWordData = True
      return True
  except FileNotFoundError:
    return False

# DATA PROCESSING

def splitFinalTestData(wordData):
    global finalTestData
    global learningData
    global splittedFinalTestData
    if splittedFinalTestData:
        return learningData, finalTestData
    else:
        for word in wordData:
            learnDataForWord, test = train_test_split(wordData[word], test_size=0.2)
            finalTestData[word] = test
            learningData[word] = learnDataForWord
        splittedFinalTestData = True
        return learningData, finalTestData

def splitTrainingTestData(learningData):
  trainTestData = {}
  for word in learningData:
      train, test = train_test_split(learningData[word], test_size=0.2)
      trainTestData[word] = {'train': train, 'test': test}
  return trainTestData

def getTrainingData(trainTestData):
  flatData = {}
  lengths = {}
  for word in trainTestData:
    trainDataForWord = trainTestData[word]['train']
    flatData[word] = numpy.concatenate(trainDataForWord)
    lengths[word] = list(map(lambda fileData: len(fileData), trainDataForWord))
  return flatData, lengths

def getTestData(trainTestData):
  labeledTestData = []
  for word in trainTestData:
    for fileData in trainTestData[word]['test']:
      labeledTestData.append((fileData, word))
  return labeledTestData

# MODEL TRAINING, SCORING

def scoreModels(testData, ghmms):
  successful = 0
  for test in testData:
    prediction = predict(test[0], ghmms)
    if prediction == test[1]:
      successful += 1
  return ( 1.0 - float(successful) / len(testData) )

def predict(fileData, ghmms):
    logOddsToKey = {}
    for key in ghmms:
        ghmm = ghmms[key]
        logOdds = ghmms[key].score_samples(fileData)[0]
        logOddsToKey[logOdds] = key
    return logOddsToKey[max(logOddsToKey.keys())]

def trainModels(trainingData, lengths, params):
  n_components, n_iter, n_mix = params
  ghmms = {}
  for word in trainingData:
    ghmm = hmm.GMMHMM(n_components=n_components, n_iter=n_iter, n_mix=n_mix)
    ghmm.fit(trainingData[word], lengths=lengths[word])
    ghmms[word] = ghmm
  return ghmms

