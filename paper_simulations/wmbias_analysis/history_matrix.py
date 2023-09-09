import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib.colors
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
from helper_make_data import make_data
from helper_mymax import mymax
from helper_history import history
from helper_func import func
import sys
# from style import style

# the axes attributes need to be set before the call to subplot
rc('font',**{'family':'sans-serif','sans-serif':['Arial']}, size=10)
rc('text', usetex=True)
rc('axes', edgecolor='black', linewidth=0.5)
rc('legend', frameon=False)
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # \boldmath

# make a custom colormap
norm=plt.Normalize(0,1)
mycmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["indigo","white"])

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
trialsback=1

SimulationName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_eps%.2f_ITI%.1f"%(AHtoWM,tauhH,tauthetaH,DH,eps,ITI)

fig, axs = plt.subplots(1,1,figsize=(1.75,1.5))


stimulus_set, stimuli, labels, readout, delay, _ =  make_data(SimulationName, N, num_sims, psycholabel)

# define a mask for the delay criterion
if ISI == 'all':
	mask_delay = np.ones(len(delay), dtype=bool)
else:
	mask_delay = delay == int(ISI)

ids = np.where( mask_delay )[0]

B, H = history(trialsback, stimuli, labels, readout, delay, stimulus_set, num_sims, num_trials, num_stimpairs, ids=ids)

im=axs.imshow(H[:num_stimpairs,:num_stimpairs], cmap=mycmap, vmin=0, vmax=1)
axs.tick_params(axis='x', direction='out')
axs.tick_params(axis='y', direction='out')
plt.colorbar(im, ax=axs, shrink=0.9, ticks=[0,0.5,1])

axs.set_xticks(np.arange(num_stimpairs))
axs.set_xticklabels(['0.3,0.2','','','','','0.8,0.7','0.2,0.3','','','','','0.7,0.8'],  rotation=45, ha='right')
#axs.set_xticklabels(['%.1f,%.1f'%(stimulus_set[i,0], stimulus_set[i,1] ) for i in range(num_stimpairs) ] ,  rotation=90)

axs.set_yticks(np.arange(num_stimpairs))
axs.set_yticklabels(['0.3,0.2','','','','','0.8,0.7','0.2,0.3','','','','','0.7,0.8'])

axs.set_xlabel("Current trial pair ($s_1, s_2$)")
axs.set_ylabel("Previous trial pair ($s_1, s_2$)")
fig.savefig("figs/history_%s_ISI%s.png"%(SimulationName,ISI), bbox_inches='tight')
fig.savefig("figs/history_%s_ISI%s.svg"%(SimulationName,ISI), bbox_inches='tight')
