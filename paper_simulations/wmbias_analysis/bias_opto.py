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
from helper_make_data import make_data
from helper_mymax import mymax
from helper_history import history
from helper_func import func
import sys

# the axes attributes need to be set before the call to subplot
rc('font',**{'family':'sans-serif','sans-serif':['Arial']}, size=8)
rc('text', usetex=True)
rc('axes', edgecolor='black', linewidth=0.5)
rc('legend', frameon=False)
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # \boldmath

tauhH=float(sys.argv[1])
tauthetaH=float(sys.argv[2])
DH=float(sys.argv[3])
eps=float(sys.argv[4])
N=int(sys.argv[5])
num_sims=int(sys.argv[6])
ISI=sys.argv[7]
ITI=int(sys.argv[8])
psycholabel=sys.argv[9] == "True"

num_stimpairs=12
num_trials=1000
trialsback=1
AHtoWMvals=[0.5,0.3]
colors=['black', cp.orange]
lsvals=['-','--']

fig, axs = plt.subplots(1,1,figsize=(1,1))#, num=1, clear=True)

j=0
for AHtoWM in AHtoWMvals:

	SimulationName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_eps%.2f_ITI%.1f"%(AHtoWM, tauhH, tauthetaH, DH, eps, ITI)
	stimulus_set, stimuli, labels, readout, delay, _ =  make_data(SimulationName, N, num_sims, psycholabel)

	if ISI == 'all':
		mask_delay = np.ones(len(delay), dtype=bool)
	else:
		mask_delay = delay == int(ISI)
	ids = np.where( mask_delay )[0]	

	B, H=history(trialsback, stimuli, labels, readout, delay, stimulus_set, num_sims, num_trials, num_stimpairs, ids=ids)

	# plot the mean
	xdata=np.arange(int(num_stimpairs/2))
	bias=np.mean(B[:,:]*100, axis=1)
	popt, pcov = curve_fit(func, xdata, bias)
	axs.scatter(xdata, bias, color=colors[j], s=5)
	axs.plot(xdata, func(xdata, popt[0], popt[1]), c=colors[j])

	# plot individual curves
	# for i in range(int(num_stimpairs/2)):
	# 	ydata=B[0,:,i]*100
	# 	axs.scatter(xdata, ydata, alpha=0.5, s=5)
	# 	popt, pcov = curve_fit(func, xdata, ydata)
	# 	axs.plot(xdata, func(xdata, popt[0], popt[1]), alpha=0.3)

	axs.set_xticks(np.arange(0,6)) 
	axs.set_xticklabels(['%.1f,%.1f'%(stimulus_set[i,0], stimulus_set[i,1] ) for i in range(6) ],  rotation=45)
	axs.spines['right'].set_visible(False)
	axs.spines['top'].set_visible(False)
	axs.set_ylim(-15,15)
	axs.set_ylabel("Stim 1 $<$ Stim 2 bias ($\%$)")
	axs.set_xlabel("%d trial back pair $(s_1, s_2)$"%trialsback)
	j+=1
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
fig.savefig("figs/bias_inactivation.png", bbox_inches='tight')
fig.savefig("figs/bias_inactivation.svg", bbox_inches='tight')