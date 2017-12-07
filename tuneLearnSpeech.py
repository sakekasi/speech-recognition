# coding: utf-8
import warnings
warnings.filterwarnings('ignore')
from learnSpeech import loadWordData, splitTrainingTestData, getTrainingData, trainModels, getTestData, scoreModels
from skopt import gp_minimize

def objective(params):
  n_components, n_iter, n_mix = params
  wordData = loadWordData()
  trainTestData = splitTrainingTestData(wordData)  
  trainingData, lengths = getTrainingData(trainTestData)
  ghmms = trainModels(trainingData, lengths)
  testData = getTestData(trainTestData)
  return scoreModels(testData)

space = [
            (1, 2), # n_components
            (1, 2)  # n_iter
        ]

res_gp = gp_minimize(objective, space, n_calls=10, random_state=0, n_jobs=-1)
print("Best score=%.4f" % res_gp.fun)
print("""Best parameters:
- n_components=%d
- n_iter=%d""" % (res_gp.x[0], res_gp.x[1]))



