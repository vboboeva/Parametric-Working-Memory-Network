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

def make_hist(SimulationName):

	before=np.arange(0,N)
	after=np.arange(2*N,3*N)

	mymaxH=[]
	mymaxWM=[]
	mystim=[]
	
	for sim in range(0, num_sims):
		f='data/'+SimulationName+'/VWM_sim%d.npy'%sim
		if os.path.isfile(f):

			VH=np.load('data/'+ SimulationName+"/VH_sim%d.npy"%sim )[:,before]
			VWM=np.load('data/'+ SimulationName+"/VWM_sim%d.npy"%sim )[:,before]
			stim=np.transpose(np.load("data/" + SimulationName+"/inf_sim%d.npy"%sim)[4:,:])

			for trial in range(num_trials):
				mymaxH=np.append(mymaxH, np.argmax(VH[trial,:])/N)
				mymaxWM=np.append(mymaxWM, np.argmax(VWM[trial,:])/N)
				mystim=np.append(mystim, stim[trial,:], axis=0)

	return mystim, mymaxH, mymaxWM

tauhH=float(sys.argv[1])
tauthetaH=float(sys.argv[2])
DH=float(sys.argv[3])
AHtoWM=float(sys.argv[4])
eps=float(sys.argv[5])
N=int(sys.argv[6])
num_sims=int(sys.argv[7])
ITI=int(sys.argv[8])

num_trials=1000

fig, ax = plt.subplots(1, 1, figsize=(1.5,1.5))

SimulationName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_eps%.2f_ITI%.1f"%(AHtoWM, tauhH, tauthetaH, DH, eps, ITI)
stimulus_set=np.load("data/" + SimulationName+"/stimulus_set.npy")

mystim, mymaxH, mymaxWM = make_hist(SimulationName)
np.save('data/mystim.npy', mystim)
np.save('data/mymaxPPC.npy', mymaxH)
np.save('data/mymaxWM.npy', mymaxWM)

bins=9
#ax.hist(mystim, density=True, color='black', histtype='step', bins=bins, label='Stim')
ax.hist(mymaxH, density=True, color=cp.pink, histtype='step', lw=2, bins=bins, label='PPC')
ax.hist(mymaxWM, density=True, color=cp.violet2, histtype='step',lw=1, bins=bins, label='WM')
ax.set_xlabel('Bump location')
ax.set_ylabel('Probability density $p_m$')

ax1=ax.twinx()

x=np.unique(stimulus_set)
y=[]
for stim in x:
	y+=[np.shape(np.where(mystim == stim))[1]]
ax1.bar(x, np.array(y)/len(mystim), width=0.04, color='gray', alpha=0.5, label='Stimulus')
ax1.set_ylabel('Probability $p_s$', color='gray')
ax1.tick_params(axis='y', labelcolor='gray')
#ax1.set_yticks([1,2], [1,2])#, rotation='vertical')
#ax1.set_ylim(0,2)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax.legend(loc=[1,0.6])#loc='lower center')
ax1.legend(loc=[1,0.6])#loc='lower center')
fig.savefig("figs/hist_bump_%s.png"%SimulationName, bbox_inches='tight')
fig.savefig("figs/hist_bump_%s.svg"%SimulationName, bbox_inches='tight')