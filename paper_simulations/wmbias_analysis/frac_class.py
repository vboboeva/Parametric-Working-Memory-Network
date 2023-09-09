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
from helper_make_data import make_data
from helper_mymax import mymax
from helper_frac_class import compute_frac_class
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
ITI=int(sys.argv[9])
psycholabel= sys.argv[10] == "True"

num_stimpairs=12

SimulationName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_eps%.2f_ITI%.1f"%(AHtoWM,tauhH,tauthetaH,DH,eps,ITI)

print(SimulationName)

stimulus_set, stimuli, labels, readout, delay =  make_data(SimulationName, N, num_sims, psycholabel)

performvals, frac_classvals = compute_frac_class(ISI, labels, stimuli, stimulus_set, readout, delay)

print(performvals)

fig, axs = plt.subplots(1,1,figsize=(1.75,1.5))
scat=axs.scatter(stimulus_set[:num_stimpairs,0], stimulus_set[:num_stimpairs,1], marker='s', s=30, c=frac_classvals[:num_stimpairs], cmap=plt.cm.coolwarm, vmin=0, vmax=1)

for i in range(int(num_stimpairs/2)):
	axs.text(stimulus_set[i,0]+0.05,stimulus_set[i,1]-0.15,'%d'%(performvals[i]*100))

for i in range(int(num_stimpairs/2),num_stimpairs):
	axs.text(stimulus_set[i,0]-0.20,stimulus_set[i,1]+0.05,'%d'%(performvals[i]*100))

axs.plot(np.linspace(0,1,10), np.linspace(0,1,10), color='black', lw=0.5)
axs.set_xlabel("Stimulus 1")
axs.set_ylabel("Stimulus 2")
axs.set_yticks([0,0.5,1])
axs.set_yticklabels([0,0.5,1])
plt.colorbar(scat,ax=axs,ticks=[0,0.5,1])
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)

fig.savefig("figs/cbiasold_%s_ISI%s.png"%(SimulationName, ISI), bbox_inches='tight')
fig.savefig("figs/cbiasold_%s_ISI%s.svg"%(SimulationName, ISI), bbox_inches='tight')