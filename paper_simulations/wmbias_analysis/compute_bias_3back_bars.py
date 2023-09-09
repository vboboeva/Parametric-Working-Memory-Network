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
from helper_make_data import make_data
from helper_mymax import mymax
from helper_history import history
from helper_func import func
import sys

# the axes attributes need to be set before the call to subplot
rc('font',**{'family':'sans-serif','sans-serif':['Arial']}, size=10)
rc('text', usetex=True)
rc('axes', edgecolor='black', linewidth=0.5)
rc('legend', frameon=False)
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # \boldmath


tauhH=float(sys.argv[1])
tauthetaH=float(sys.argv[2])
DH=float(sys.argv[3])
AHtoWM=float(sys.argv[4])
eps=float(sys.argv[5])
N=int(sys.argv[6])
num_sims=int(sys.argv[7])
ISI=sys.argv[8]
psycholabel=sys.argv[9] == "True"

num_trials=1000
num_stimpairs=12
trialsback=np.arange(1,9)
ITIvals=[1.2,5,10]


if ISI == 'all':
	for ITI in ITIvals:
		SimulationName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_eps%.2f_ITI%.1f"%(AHtoWM,tauhH,tauthetaH,DH,eps,ITI)
		stimulus_set, stimuli, labels, readout, delay, _ =  make_data(SimulationName, N, num_sims, psycholabel)

		B, H=history(trialsback, stimuli, labels, readout, delay, stimulus_set, num_sims, num_trials, num_stimpairs)

		slopes=np.empty(len(trialsback))
		for o in range(len(trialsback)):
			xdata=np.arange(int(num_stimpairs/2))
			bias=np.mean(B[o,:,:]*100, axis=1)
			popt, pcov = curve_fit(func, xdata, bias)
			slopes[o]=popt[0]
			print(slopes[o])
		print(np.shape(bias))
		np.save("data/attraction_repulsion_ITI%.1f"%ITI, slopes)

else:
	ISIvals=[2,6,10]
	for ISI in ISIvals:
		for ITI in ITIvals:
			SimulationName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_eps%.2f_ITI%.1f"%(AHtoWM,tauhH,tauthetaH,DH,eps,ITI)
			stimulus_set, stimuli, labels, readout, delay, _ =  make_data(SimulationName, N, num_sims, psycholabel)
			B, H=history(trialsback, stimuli, labels, readout, delay, stimulus_set, num_sims, num_trials, num_stimpairs)
			for o in range(len(trialsback)):
				xdata=np.arange(int(num_stimpairs/2))
				bias=np.mean(B[o,:,:]*100, axis=1)
				popt, pcov = curve_fit(func, xdata, bias)
				slopes[o]=popt[0]
				print(slopes[o])
			np.save("data/attraction_repulsion_ISI%.1f_ITI%.1f"%(ISI,ITI), B)

