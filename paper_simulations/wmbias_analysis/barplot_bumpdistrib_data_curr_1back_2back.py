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

trials_eliminate = np.append(np.arange(0,num_trials*num_sims,500), np.arange(1,num_trials*num_sims,500))
num_trials_consider = num_trials*num_sims - len(np.append(np.arange(0,num_trials*num_sims,500), np.arange(1,num_trials*num_sims,500)))

stimuli, VWM, VH = make_data()


VV=[VH, VWM]
ind = np.arange(8)
width = 0.3       
colors=[cp.violet2, cp.pink ]
labels=['WM', 'PPC' ]

p1=[]
p2=[]
p3=[]

stim_to_idx={0.2:0, 0.3:1, 0.4:2, 0.5:3, 0.6:4, 0.7:5, 0.8:6, 0.45:7, 0.55:8}
idx_to_stim={0:0.2, 1:0.3, 2:0.4, 3:0.5, 4:0.6, 5:0.7, 6:0.8, 7:0.45, 8:0.55}
distances=99*np.ones((len(stim_to_idx), len(stim_to_idx), num_trials_consider))

fig, axs = plt.subplots(3,3,figsize=(14,10))#, num=1, clear=True)

for i in range(1):
	print(labels[i])
	V=VV[i]
	j=0
	for trial in range(len(stimuli)):
		if trial in trials_eliminate: # remove the first two trials of every simulation since no hist effects
			continue	
		else:
			posmax=np.argmax(V[trial,:])/(1.*N)
			x = stimuli[trial-1, 1]
			y = stimuli[trial, 0]
			idx = stim_to_idx[x]
			idy = stim_to_idx[y]
			distances[idx, idy, j] = posmax
			j+=1

	curr_idx=0
	plot_idx=np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
	
	for k in range(7):
		take_idx = np.where(distances[k, curr_idx, :] != 99.)
		d = distances[k, curr_idx, take_idx]
		d = d[0,:]
		axs[plot_idx[k,0], plot_idx[k,1]].hist(d, bins=30, density=True, color=colors[i], label=labels[i])
		axs[plot_idx[k,0], plot_idx[k,1]].set_xlim(0,1)
		axs[plot_idx[k,0], plot_idx[k,1]].set_ylim(0,10)
		axs[plot_idx[k,0], plot_idx[k,1]].set_title('s2(t-1)=%.1f s1(t)=%.1f'%(idx_to_stim[k], idx_to_stim[curr_idx])) 
		axs[plot_idx[k,0], plot_idx[k,1]].axvline(idx_to_stim[k], color='red')
		axs[plot_idx[k,0], plot_idx[k,1]].axvline(idx_to_stim[curr_idx], color='green')
	axs[2,0].set_xlabel('Bump position before s2(t)')
	axs[2,0].set_ylabel('Pdf')

fig.savefig("figs/s1t%s_%s.png"%(idx_to_stim[curr_idx], SimulationName), bbox_inches='tight')
fig.savefig("figs/s1t%s_%s.svg"%(idx_to_stim[curr_idx], SimulationName), bbox_inches='tight')

