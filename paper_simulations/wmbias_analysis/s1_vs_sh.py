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
import color_palette as cp

# the axes attributes need to be set before the call to subplot
rc('font',**{'family':'sans-serif','sans-serif':['Arial']}, size=10)
rc('text', usetex=True)
rc('axes', edgecolor='black', linewidth=0.5)
rc('legend', frameon=False)
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # \boldmath

AHtoWM=0.5#float(sys.argv[4])
tauhH=0.5#float(sys.argv[1])
tauthetaH=7.5#float(sys.argv[2])
ISI=2
N=2000
num_sims=50
num_trials=1000
num_stimpairs=12
trialback=1
DH=0.3
vicinity=0.05
eps=0
ITI=5

# LINEAR FIT 
def func(xvals,a,b):	
	return a*xvals+b

def make_data():
	stimulus_set=np.load( 'data/'+SimulationName+"/stimulus_set.npy")
	dstim_set=np.load( 'data/'+SimulationName+"/dstim_set.npy")			

	stimuli=np.empty((0,2), float)
	VWM=np.empty((0,N), float)
	VH=np.empty((0,N), float)
	takevalsWM=np.arange(0,N) 
	takevalsH=np.arange(0,N)

	for sim in range(num_sims):
		myfile='data/'+SimulationName+'/VWM_sim%d.npy'%sim
		if os.path.isfile(myfile):
			# print(myfile)
			stimuli=np.append(stimuli, np.transpose(np.load("data/" + SimulationName+"/inf_sim%d.npy"%sim)[4:,:]), axis=0)
			VWM=np.append(VWM, np.load( 'data/'+SimulationName+"/VWM_sim%d.npy"%sim)[:,takevalsWM], axis=0)
			VH=np.append(VH, np.load( 'data/'+SimulationName+"/VH_sim%d.npy"%sim)[:,takevalsH], axis=0)

	return stimuli, VWM, VH	


SimulationName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_eps%.2f_ITI%.1f"%(AHtoWM, tauhH, tauthetaH, DH, eps, ITI)

stimuli, V, VH = make_data()

stim_to_idx={0.20:0, 0.30:1, 0.40:2, 0.45:3, 0.50:4, 0.55:5, 0.60:6, 0.70:7, 0.80:8}

stimvals=[0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]

loc=99*np.ones(( len(stim_to_idx), num_sims*num_trials ))

fig, axs = plt.subplots(1,1,figsize=(2,2))

print(stimuli)
j=0
for trial in range(len(stimuli)):
	posmax=np.argmax(V[trial,:])/(1.*N)
	s1 = stimuli[trial, 0]
	loc[stim_to_idx[s1], j] = posmax
	j+=1

for k, stim in enumerate(stimvals):
	take_idx = np.where(loc[k, :] != 99.)
	axs.scatter(np.repeat(stim, np.shape(take_idx)[1]), loc[k,take_idx], alpha=0.05, color='gray', s=0.5)
	axs.scatter(stim, np.mean(loc[k,take_idx]), color='k')

axs.plot(np.arange(0.1,1,0.1), np.arange(0.1,1,0.1), ls='--', color='k')
axs.set_xlim(0,1)
axs.set_ylim(0,1)
# axs.legend(loc='best')
axs.set_xlabel('s1(t)')
axs.set_ylabel('sh(t)')

fig.savefig("figs/s1_vs_sh_%s.png"%(SimulationName), bbox_inches='tight')
fig.savefig("figs/s1_vs_sh_%s.svg"%(SimulationName), bbox_inches='tight')

