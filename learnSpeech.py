import pickle

from hmmlearn import hmm

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plot
import numpy

from sklearn.model_selection import train_test_split

WORDDATA='worddata.p'
# SPEECHMODELS='speechmodels.p'

wordData = {}
loadedWordData = False

# FILE PROCESSING

def loadWordData():
  if loadedWordData:
    return wordData
  else if deserializeWordData():
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
  try:
    with open(WORDDATA, 'rb') as wordDataFile:
      wordData = pickle.load(WORDDATA)
      loadedWordData = True
      return true
  except FileNotFoundError:
    return false

# DATA PROCESSING

def splitTrainingTestData(wordData):
  trainTestData = {}
  for word in wordData:
      train, test = train_test_split(wordData[word], test_size=0.3)
      trainTestData[word] = {'train': train, 'test': test}
  return trainTestData

def getTrainingData(trainTestData):
  flatData = {}
  lengths = {}
  for word in wordData:
    flatDataForWord = []
    runLengthsForWord = []
    for fileData in trainTestData[directory]['train']:
      runLengthsForWord.append(len(fileData))
      flatDataForWord += fileData.flatten().tolist()
      
      flatData[directory] = numpy.array(flatDataForWord).reshape(-1, 13)
      lengths[directory] = runLengthsForWord
  return flatData, lengths

def getTestData(trainTestData):
  labeledTestData = []
  for word in trainTestData:
    for fileData in data[word]['test']:
      labeledTestData.append((fileData, key))
  return labeledTestData

# MODEL TRAINING, SCORING

def scoreModels(testData):
  successful = 0
  for test in testData:
    prediction = predict(test[0])
    if prediction == test[1]:
      successful += 1
  return ( 1.0 - float(successful) / len(labeledTestData) )

def predict(fileData):
    logOddsToKey = {}
    for key in ghmms:
        ghmm = ghmms[key]
        logOdds = ghmms[key].score_samples(fileData)[0]
        logOddsToKey[logOdds] = key
    return logOddsToKey[max(logOddsToKey.keys())]

def trainModels(trainingData, lengths):
  ghmms = {}
  for word in trainingData:
    ghmm = hmm.GMMHMM(n_components=n_components, n_iter=n_iter, n_mix=n_mix)
    ghmm.fit(trainingData[word], lengths=lengths[word])
    ghmms[word] = ghmm
  return ghmms

