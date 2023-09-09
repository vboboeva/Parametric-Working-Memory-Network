import numpy as np
import random
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
import pandas
from pandas import DataFrame
import numpy_indexed as npi
from matplotlib import rc
from pylab import rcParams
from scipy.optimize import curve_fit
import color_palette as cp
from helper_func import func
# from helper_make_data import make_data
# from helper_mymax import mymax
# from helper_history import history

def history(trialsback, stimuli, labels, readout, delay, stimulus_set, num_sims, num_trials, num_stimpairs, ids=None):

	if ids is None:
		ids = np.arange(len(stimuli))

	ids_shifted = ids - trialsback
	_mask = ids_shifted > -1
	ids = ids[_mask]
	ids_shifted = ids_shifted[_mask]

	stims = stimuli[ids]
	stims_shifted = stimuli[ids_shifted]

	trialtypevals=np.zeros((len(stimulus_set), len(stimulus_set)))
	responsevals=np.zeros((len(stimulus_set), len(stimulus_set)))

	#print(np.arange(0, num_sims*num_trials, num_trials))
	# SORT performance by previous pair of stimuli

	for idx, stim, stim_s in zip(ids, stims, stims_shifted):
		for m in range(len(stimulus_set)):
			if ( stim == stimulus_set[m] ).all():
				for n in range(len(stimulus_set)):
					if ( stim_s == stimulus_set[n] ).all():
						trialtypevals[n,m] += 1
						responsevals[n,m] += readout[idx]

	B=np.empty((int(num_stimpairs/2), int(num_stimpairs/2)))
	A11=responsevals[:int(num_stimpairs/2),:int(num_stimpairs/2)]/trialtypevals[:int(num_stimpairs/2),:int(num_stimpairs/2)]
	B11=np.zeros((int(num_stimpairs/2),int(num_stimpairs/2)))
	for i in range(int(num_stimpairs/2)):
		B11[:,i] = A11[:,i] - np.mean(A11[:,i])
	B[:,:]=B11
	H=np.divide(responsevals, trialtypevals, out=np.zeros_like(responsevals), where=trialtypevals!=0.)
	return B, H
