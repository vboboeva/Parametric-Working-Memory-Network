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
	delay=np.empty((0,1), float)
	stimuli=np.empty((0,2), float)
	VWM=np.empty((0,N), float)
	VH=np.empty((0,N), float)
	takevalsWM=np.arange(0,N) 
	takevalsH=np.arange(0,N)

	for sim in range(num_sims):
		myfile='data/'+SimulationName+'/VWM_sim%d.npy'%sim
		if os.path.isfile(myfile):
			# print(myfile)
			delay=np.append(delay, np.load("data/" + SimulationName+"/inf_sim%d.npy"%sim)[0,:])
			stimuli=np.append(stimuli, np.transpose(np.load("data/" + SimulationName+"/inf_sim%d.npy"%sim)[4:,:]), axis=0)
			VWM=np.append(VWM, np.load( 'data/'+SimulationName+"/VWM_sim%d.npy"%sim)[:,takevalsWM], axis=0)
			VH=np.append(VH, np.load( 'data/'+SimulationName+"/VH_sim%d.npy"%sim)[:,takevalsH], axis=0)

	return stimuli, VWM, delay	


SimulationName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_eps%.2f_ITI%.1f"%(AHtoWM, tauhH, tauthetaH, DH, eps, ITI)

mask_trials = np.array([i for i in np.arange(num_trials*num_sims) if i not in np.arange(0,num_trials*num_sims, num_trials)])

print(mask_trials)

num_trials_consider = num_trials*num_sims - len(np.arange(0,num_trials*num_sims,num_trials))

stimuli, V, delay = make_data()


colorvals=[cp.green1, cp.green2, cp.green3, 'black']
delayvals=[2,6,10]
lsvals=['-','-','-','--']

diststim_to_idx={-0.6:0, -0.5:1, -0.4:2, -0.3:3, -0.2:4, -0.1:5, -0.05:6, 0:7, 0.05:8, 0.1:9, 0.2:10, 0.3:11, 0.4:12, 0.5:13, 0.6:14}
distances=99*np.ones(( len(diststim_to_idx), 3, num_trials_consider ))

fig, axs = plt.subplots(1,1,figsize=(2,2))

for i, ISI in enumerate(delayvals):

	print(ISI)
	# take those with a given ISI
	if ISI !='all':
		mask_delay = np.where(delay == ISI)[0]
	else:
		mask_delay = np.arange(num_trials*num_sims) 

	ids = list(set(mask_delay).intersection(mask_trials))
	print(ids[:20])
	j=0
	for t, trial in enumerate(ids):
		posmax=np.argmax(V[trial,:])/(1.*N)
		z = stimuli[trial-1, 0]
		x = stimuli[trial-1, 1]
		y = stimuli[trial, 0]

		d = round(y - x, 4)
		if d in diststim_to_idx:
			distances[diststim_to_idx[d], 0, j] = d
			distances[diststim_to_idx[d], 1, j] = y - posmax
			distances[diststim_to_idx[d], 2, j] = z - posmax
			j+=1

	mean_distances=np.ndarray(( len(diststim_to_idx), 3 ))

	for k in range(len(diststim_to_idx)):
		take_idx = np.where(distances[k, 0, :] != 99.)
		mean_distances[k,0] = np.nanmean(distances[k, 0, take_idx])
	
		take_idx = np.where(distances[k, 1, :] != 99.)
		mean_distances[k,1] = np.nanmean(distances[k, 1, take_idx])
	
		take_idx = np.where(distances[k, 2, :] != 99.)
		mean_distances[k,2] = np.nanmean(distances[k, 2, take_idx])

	axs.plot(np.arange(-0.6,0.6,0.1), np.arange(-0.6,0.6,0.1), ls=':', color='k')
	axs.plot(mean_distances[:,0], mean_distances[:,1], color=colorvals[i], label='%s'%ISI)
	axs.axhline(0, color='black')
	axs.axvline(0, color='black')
	axs.set_xlim(-0.6,0.6)
	axs.set_ylim(-0.2, 0.2)
	axs.legend(loc='lower right')
	axs.set_xlabel('s1(t)-s2(t-1)')
	axs.set_ylabel('s1(t)-sh(t)')

fig.savefig("figs/displ_ISI_%s.png"%(SimulationName), bbox_inches='tight')
fig.savefig("figs/displ_ISI_%s.svg"%(SimulationName), bbox_inches='tight')

