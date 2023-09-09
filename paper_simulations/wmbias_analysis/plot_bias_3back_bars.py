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

if ISI == 'all':
	ITIvals=[1.2,5,10]
	colors=[cp.violet1, cp.violet2, cp.violet3]
	fig, axs = plt.subplots(1,1,figsize=(3,1))
	width=0.3
	SimulationName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_eps%.2f"%(AHtoWM,tauhH,tauthetaH,DH,eps)
	for i in range(len(ITIvals)):
		A=np.load("data/attraction_repulsion_ITI%.1f.npy"%ITIvals[i])
		axs.bar(trialsback+(i-1)*width*np.ones(len(trialsback)), A, width=width, color=colors[i], label='%.1f'%(ITIvals[i]+1.)) # +1 because there is +1s at start of trial
	axs.set_xlabel("Trials back")
	axs.set_ylabel("Slope of bias")
	axs.set_xticks(trialsback) 
	axs.spines['right'].set_visible(False)
	axs.spines['top'].set_visible(False)
	plt.legend()
	fig.savefig("figs/bias_bars_%s.png"%(SimulationName), bbox_inches='tight')
	fig.savefig("figs/bias_bars_%s.svg"%(SimulationName), bbox_inches='tight')	

else:
	fig, axs = plt.subplots(3,1,figsize=(3,4.5))
	ISIvals=[2,6,10]
	j=0
	for ISI in ISIvals:
		ITIvals=[1.2,5,10]
		colors=[cp.violet1, cp.violet2, cp.violet3]
		width=0.3
		SimulationName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_eps%.2f"%(AHtoWM,tauhH,tauthetaH,DH,eps)
		for i in range(len(ITIvals)):
			A=np.load("data/attraction_repulsion_ISI%.1f_ITI%.1f.npy"%(ISI,ITIvals[i]))
			axs[j].bar(trialsback+(i-1)*width*np.ones(len(trialsback)), A, width=width, color=colors[i], label='%.1f'%(ITIvals[i]+1.)) # +1 because there is +1s at start of trial
		axs[j].set_xlabel("Trials back")
		axs[j].set_ylabel("Slope of bias")
		axs[j].set_xticks(trialsback) 
		axs[j].spines['right'].set_visible(False)
		axs[j].spines['top'].set_visible(False)
		axs[j].set_ylim(-5,1)
		j+=1
	plt.legend()
	fig.savefig("figs/bias_bars_%s_diffISI.png"%(SimulationName), bbox_inches='tight')
	fig.savefig("figs/bias_bars_%s_diffISI.svg"%(SimulationName), bbox_inches='tight')	

