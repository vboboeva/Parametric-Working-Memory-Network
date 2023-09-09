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
from helper_psycho import psychometric
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
eps=float(sys.argv[4])
N=int(sys.argv[5])
num_sims=int(sys.argv[6])
ISI=sys.argv[7]
ITI=int(sys.argv[8])
psycholabel=sys.argv[9] == "True"

num_stimpairs=12
AHtoWMvals=[0.5,0.3]
colors=['black', cp.orange]
lsvals=['-','--']
curvelabels=['intact','reduced']

fig, axs = plt.subplots(1,1,figsize=(1.5,1.5))
j=0
for AHtoWM in AHtoWMvals:
	SimulationName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_eps%.2f_ITI%.1f"%(AHtoWM,tauhH,tauthetaH,DH,eps,ITI)
	stimulus_set, stimuli, labels, readout, delay, _ =  make_data(SimulationName, N, num_sims, psycholabel)
	dstim_set, psy, sd, numpts = psychometric(SimulationName, stimuli, readout, delay, ISI)
	print(psy)
	print(sd)
	axs.errorbar(dstim_set, psy, yerr=sd/np.sqrt(numpts), color=colors[j], label='%s'%curvelabels[j])
	j+=1

axs.set_xlabel("Stimulus 1 - Stimulus 2")
axs.set_ylabel("Fraction classified \n Stim 1 $>$ Stim 2")
axs.set_ylim(0,1.1)
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
plt.legend() #, prop=fontP
fig.savefig("figs/psycho_both_%s.png"%(SimulationName), bbox_inches='tight')
fig.savefig("figs/psycho_both_%s.svg"%(SimulationName), bbox_inches='tight')