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

ghmms = pickle.load(open("hmmset.p", "rb"))
for key in ghmms:
    print("Model for " + key + ":")
    print("Log Odds: " + ghmm[key].score(features))

