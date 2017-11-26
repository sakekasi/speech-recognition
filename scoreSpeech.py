from os import listdir
from os.path import isfile, join
import pickle
import argparse

from hmmlearn import hmm

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plot
import numpy

parser = argparse.ArgumentParser(description='Get the log odds of audio being generated from hmm of specific word')
parser.add_argument('datadir')
parser.add_argument('file')
args = parser.parse_args()

filepath = "data/" + args.datadir + "/" + args.file
print(filepath)
(rate, signal) = wav.read(filepath)
features = mfcc(signal, rate, winfunc=numpy.hamming)

ghmmInfo = pickle.load(open("hmmset.p", "rb"))
ghmms = {}
for key in ghmmInfo:
    i = ghmmInfo[key]
    n_features = i[0]
    transmat = i[1]
    startprob = i[2]
    means = i[3]
    covars = i[4]
    ghmms[key] = hmm.GaussianHMM(startprob_prior=startprob, 
                                 transmat_prior=transmat,
                                 means_prior=means,
                                 covars_prior=covars, 
                                 init_params='stmc')

for key in ghmms:
    print("Model for " + key + ":")
    print("Log Odds: " + ghmms[key].score_samples(features))

