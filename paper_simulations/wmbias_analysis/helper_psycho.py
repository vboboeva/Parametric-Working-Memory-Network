import numpy as np
import random
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression, SGDClassifier
import pandas
from pandas import DataFrame
import numpy_indexed as npi
from matplotlib import rc
from pylab import rcParams
from scipy.optimize import curve_fit
import color_palette as cp

def psychometric(SimulationName, stimuli, readout, delay, ISI):
	dstim_set=np.load("data/" + SimulationName+"/dstim_set.npy")			
	if ISI != 'all':
		# take those with a given ISI
		mask0 = np.where(delay == ISI)
		readout = readout[mask0]
		stimuli = stimuli[mask0]

	# take only psychometric stimuli (where stim2=0.5 is fixed and stim 1 is varied)
	mask=np.where(stimuli[:,1]==0.5)
	readout1=readout[mask]
	stimuli1=stimuli[mask] 

	dstim=(stimuli1[:,0]-stimuli1[:,1])#.round(decimals=4)
	psy=np.zeros(len(dstim_set))
	sd=np.zeros(len(dstim_set))
	numpts=np.zeros(len(dstim_set))
	k=0
	for j in dstim_set:
		psy[k]=np.nanmean(readout1[np.where(np.isclose(dstim, j, atol=1e-1))])
		sd[k]=np.std(readout1[np.where(np.isclose(dstim, j, atol=1e-1))])
		numpts[k]=len(readout1[np.where(np.isclose(dstim, j, atol=1e-1))])
		k+=1

	return dstim_set, psy, sd, numpts
