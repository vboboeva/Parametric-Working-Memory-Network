import numpy as np
import random
import os
import sys
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
import color_palette as cp
from helper_make_data import make_data
from helper_mymax import mymax
from helper_frac_class import compute_frac_class
from helper_func import func

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
ITI=int(sys.argv[8])
psycholabel=sys.argv[9] == "True"


num_stimpairs=12
ISIvals=[2,6,10]

stimulus_biasplus=[[0.3,0.2],[[0.4,0.3]],[0.6,0.7],[0.7,0.8]]
stimulus_biasminus=[[0.6,0.5],[0.7,0.6],[0.8,0.7],[0.2,0.3],[0.3,0.4],[0.4,0.5]] # ,[0.5,0.6], [0.5,0.4] 

indices_biasplus=[0,1,10,11]
indices_biasminus=[3,4,5,6,7,8]

pos=[]
neg=[]

SimulationName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_eps%.2f_ITI%.1f"%(AHtoWM,tauhH,tauthetaH,DH,eps,ITI)

stimulus_set, stimuli, labels, readout, delay, _ =  make_data(SimulationName, N, num_sims, psycholabel)
for ISI in ISIvals:
	print(ISI)
	performvals, scattervals = compute_frac_class(ISI, labels, stimuli, stimulus_set, readout, delay)
	pos+=[np.nanmean(performvals[indices_biasplus])-np.nanmean(performvals[:num_stimpairs])]
	neg+=[np.nanmean(performvals[indices_biasminus])-np.nanmean(performvals[:num_stimpairs])]

print(ISIvals)
print(pos)
# print(neg)

fig, axs = plt.subplots(1,1,figsize=(1,1))

axs.bar([1,3,5],[pos[i]*100 for i in range(len(pos))], color=cp.green)
axs.bar([1,3,5],[neg[i]*100 for i in range(len(neg))], color=cp.orange)
# axs.set_ylim(-15,15)
# axs.set_xticks(ISIvals)
axs.set_xticklabels(ISIvals)
axs.axhline(0,color='k')
axs.set_xlabel("Delay interval [s]")
axs.set_ylabel("$\%$ correct minus average")
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
fig.savefig('figs/cbias_fISI_%s.png'%SimulationName,bbox_inches='tight')
fig.savefig('figs/cbias_fISI_%s.svg'%SimulationName,bbox_inches='tight')


