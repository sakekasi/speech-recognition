from os import listdir
from os.path import isfile, join
import pickle

from hmmlearn import hmm

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plot
import numpy

ghmms = {}

datadirs = [f for f in listdir("data/")]
for directory in datadirs:
    curdir = "data/" + directory
    datafiles = [f for f in listdir(curdir) if isfile(join(curdir,f))]
    ghmm = hmm.GaussianHMM(n_components=10)

    for f in datafiles:
        (rate, signal) = wav.read(curdir + "/" + f)
        ghmm.fit(mfcc(signal, rate, winfunc=numpy.hamming))

    ghmms[directory] = (ghmm.n_features, ghmm.transmat_, 
                ghmm.startprob_, ghmm.means_, ghmm.covars_)

pickle.dump(ghmms, open("hmmset.p", "wb"))
