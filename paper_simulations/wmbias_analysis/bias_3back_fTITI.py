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
import matplotlib as mpl
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
trialsback=np.arange(1,4)

ITIvals=[1.2,5,10]
colorvals=[cp.violet1, cp.violet2, cp.violet3]
lsvals=['-','-','-']


fig, axs = plt.subplots(1,3,figsize=(5,1.5))

# First plot the bias for each ITI separately
for k, ITI in enumerate(ITIvals):
	
	SimulationName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_eps%.2f_ITI%.1f"%(AHtoWM,tauhH,tauthetaH,DH,eps,ITI)
	stimulus_set, stimuli, labels, readout, delay, _ =  make_data(SimulationName, N, num_sims, psycholabel)

	for o, back in enumerate(trialsback):
		# define a mask for the delay criterion
		if ISI == 'all':
			mask_delay = np.ones(len(delay), dtype=bool)
		else:
			mask_delay = delay == int(ISI)

		ids = np.where( mask_delay )[0]

		B, H=history(back, stimuli, labels, readout, delay, stimulus_set, num_sims, num_trials, num_stimpairs, ids=ids)

		# plot the mean
		xdata=np.arange(int(num_stimpairs/2))
		bias=np.mean(B[:,:]*100, axis=1)
		popt, pcov = curve_fit(func, xdata, bias)
		axs[o].scatter(xdata, bias, c=colorvals[k], s=5)
		axs[o].plot(xdata, func(xdata, popt[0], popt[1]), c=colorvals[k], ls=lsvals[k])

		# # plot individual curves
		# for i in range(int(num_stimpairs/2)):
		# 	ydata=B[:,i]*100
		# 	# axs[o].scatter(xdata, ydata, alpha=0.5, s=5)
		# 	popt, pcov = curve_fit(func, xdata, ydata)
		# 	axs[o].plot(xdata, func(xdata, popt[0], popt[1]), alpha=0.3)

		axs[o].set_xticks(np.arange(0,6)) 
		axs[o].set_xticklabels(['%.1f,%.1f'%(stimulus_set[i,0], stimulus_set[i,1] ) for i in range(6) ],  rotation=45)
		axs[o].spines['right'].set_visible(False)
		axs[o].spines['top'].set_visible(False)
		axs[o].set_ylim(-21,21)
		if o==0:
			axs[o].set_ylabel("Stim 1 $<$ Stim 2 bias ($\%$)")
			axs[o].set_xlabel("%d trial back pair $(s_1, s_2)$"%trialsback[o])
		else:
			axs[o].set_xticklabels([])
			axs[o].set_yticklabels([])
			axs[o].set_xlabel("%d trials back"%trialsback[o])

fig.savefig("figs/bias_%s_fITI.png"%(SimulationName), bbox_inches='tight')
fig.savefig("figs/bias_%s_fITI.svg"%(SimulationName), bbox_inches='tight')
