import numpy as np
import random
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression, SGDClassifier, LinearRegression
from sklearn.ensemble import RandomForestRegressor as RF
import pandas
from pandas import DataFrame
import numpy_indexed as npi
from matplotlib import rc
from pylab import rcParams
from scipy.optimize import curve_fit
# from helper_mymax import mymax
from helper_make_data import make_data
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
ITI=int(sys.argv[9])
psycholabel=sys.argv[10] == "True"

num_stimpairs=12
num_trials=1000
trialback=1

vicinity=0.05
width=0.02

# LINEAR FIT 
def func(xvals,a,b):	
	return a*xvals+b

def find_trials(stimuli, SimulationName, VH):
	
	trials_jumps=np.empty((0,1), int)
	trials_nojumps=np.empty((0,1), int)

	dist=[]
	for i in range(len(stimuli)):
		posmax=VHmax[i]
		if (i>0):
			dist=np.append(dist, np.abs(posmax-posmax_prev))

		# no jump condition
		if (i>0 and np.abs(posmax-posmax_prev) <= width):
			trials_nojumps=np.append(trials_nojumps, i)
		
		# jump condition
		elif (i>0 and np.abs(posmax-posmax_prev) > width):
			trials_jumps=np.append(trials_jumps, i)
	
		posmax_prev=posmax

	print("jumps=%d, nojumps=%d"%(len(trials_jumps), len(trials_nojumps)))
	return trials_jumps, trials_nojumps

def history(trialback, stimuli, trials_jumps):
	stimulus_set=np.load("data/"+SimulationName+"/stimulus_set.npy")

	trialtypevals=np.zeros((len(stimulus_set), len(stimulus_set)))
	responsevals=np.zeros((len(stimulus_set), len(stimulus_set)))

	# SORT performance by previous pair of stimuli
	for idx in range(len(stimuli)):
		if idx in trials_jumps:
			for m in range(len(stimulus_set)):
				if ( stimuli[idx]==stimulus_set[m] ).all():
					for n in range(len(stimulus_set)):
						if ( stimuli[idx-trialback]==stimulus_set[n] ).all():
							trialtypevals[n,m] += 1
							responsevals[n,m] += readout[idx]

	A11=responsevals[:int(num_stimpairs/2),:int(num_stimpairs/2)]/trialtypevals[:int(num_stimpairs/2),:int(num_stimpairs/2)]
	B11=np.zeros((int(num_stimpairs/2),int(num_stimpairs/2)))
	for i in range(int(num_stimpairs/2)):
		B11[:,i] = A11[:,i] - np.nanmean(A11[:,i])

	H=np.divide(responsevals, trialtypevals, out=np.zeros_like(responsevals), where=trialtypevals!=0)
	return B11, H

SimulationName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_eps%.2f_ITI%.1f"%(AHtoWM, tauhH, tauthetaH, DH, eps, ITI)

stimulus_set, stimuli, labels, readout, delay, VHmax = make_data(SimulationName, N, num_sims, psycholabel)

trials_jumps, trials_nojumps = find_trials(stimuli, SimulationName, VHmax)

B_j, H_j=history(trialback, stimuli, trials_jumps)
B_nj, H_nj=history(trialback, stimuli, trials_nojumps)


fig, axs = plt.subplots(1,1,figsize=(1.5,1.5)) #, num=1, clear=True)
xdata=np.arange(int(num_stimpairs/2))

for i in range(int(num_stimpairs/2)):
	ydata=B_j[:,i]*100
	#axs.scatter(xdata, ydata, alpha=0.5, s=5)
	popt, pcov = curve_fit(func, xdata, ydata)
	axs.plot(xdata, func(xdata, popt[0], popt[1]), alpha=0.3)

bias_j=np.mean(B_j*100, axis=1)
popt, pcov = curve_fit(func, xdata, bias_j)

axs.scatter(xdata, bias_j, color='black', s=10, marker='^')
axs.plot(xdata, func(xdata, popt[0], popt[1]), color='black', label='Jump prev. trial')

for i in range(int(num_stimpairs/2)):
	ydata=B_nj[:,i]*100
	#axs.scatter(xdata, ydata, alpha=0.5, s=5)
	popt, pcov = curve_fit(func, xdata, ydata)
	axs.plot(xdata, func(xdata, popt[0], popt[1]), ls='--', alpha=0.3)

bias_nj=np.mean(B_nj*100, axis=1)
popt, pcov = curve_fit(func, xdata, bias_nj)

axs.scatter(xdata, bias_nj, color='black', s=10, marker='o')
axs.plot(xdata, func(xdata, popt[0], popt[1]), ls='--', color='black', label='No jump prev. trial')

#axs.legend(ncol=2)
axs.set_xticks(np.arange(0,6)) 
axs.set_xticklabels(['%.1f,%.1f'%(stimulus_set[i,0], stimulus_set[i,1] ) for i in range(6) ],  rotation=30)
#axs.set_ylim(-20,20)
axs.set_xlabel("%d trial back pair $(s_1, s_2)$"%trialback)
axs.set_ylabel("Bias stim. 1 $<$ stim. 2 ($\%$)")
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
fig.legend(loc='upper right') #, prop=fontP
fig.savefig("figs/bias_jump_%s.png"%(SimulationName), bbox_inches='tight')
fig.savefig("figs/bias_jump_%s.svg"%(SimulationName), bbox_inches='tight')