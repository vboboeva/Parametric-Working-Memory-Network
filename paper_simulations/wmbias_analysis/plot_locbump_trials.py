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

tauhH=float(sys.argv[1])
tauthetaH=float(sys.argv[2])
DH=float(sys.argv[3])
AHtoWM=float(sys.argv[4])
eps=float(sys.argv[5])
N=int(sys.argv[6])
num_sims=int(sys.argv[7])
ISI=sys.argv[8]
ITI=int(sys.argv[9])

num_trials=500

SimulationName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_eps%.2f_ITI%.1f"%(AHtoWM,tauhH,tauthetaH,DH,eps,ITI)

before=np.arange(0,N)
steps=num_trials

for sim in range(0,1):
	
	VH=np.load("data/"+SimulationName+"/VH_sim%d.npy"%sim )[:,before]
	VWM=np.load("data/"+SimulationName+"/VWM_sim%d.npy"%sim )[:,before]
	stim=np.transpose(np.load("data/"+SimulationName+"/inf_sim%d.npy"%sim )[4:6,:])
	print(stim)
	delay=np.load("data/" + SimulationName+"/inf_sim%d.npy"%sim)[0,:]

	print(np.shape(stim))
	print(np.shape(VWM))
	mymaxVH=[]
	mymaxVWM=[]
	meanstim=[]
	
	for trial in range(num_trials):
		mymaxVH=np.append(mymaxVH,np.argmax(VH[trial,:])/N)
		mymaxVWM=np.append(mymaxVWM,np.argmax(VWM[trial,:])/N)
		meanstim=np.append(meanstim,np.mean(stim[:trial,:]))

	fig, ax1 = plt.subplots(1,1,figsize=(6,1.5))
	for step in range(steps):
		if delay[step] == 2:
			p1=ax1.scatter(step, stim[step,0], color=cp.green, s=10, label='Stim. 1, delay=$2s$')
		elif delay[step] == 6:
			p2=ax1.scatter(step, stim[step,0], color=cp.red, s=10, label='Stim. 1, delay=$6s$')
		elif delay[step] == 10:
			p3=ax1.scatter(step, stim[step,0], color=cp.blue, s=10, label='Stim. 1, delay=$10s$')

	p4,=ax1.plot(np.arange(steps), meanstim[:steps],'black', label='Running mean stimulus')
	p5,=ax1.plot(np.arange(steps), mymaxVH[:steps], lw=2, label='PPC', color=cp.pink)
	p6,=ax1.plot(np.arange(steps), mymaxVWM[:steps], lw=1, label='WM', color=cp.violet2)
	ax1.set_xlim(0,50)
	ax1.set_ylim(0,1)
	#ax1.set_xticks([0,50,steps])
	#ax1.set_xticklabels([0,50,steps])
	ax1.set_xlabel('Trial')
	ax1.set_ylabel('Bump location \n before stim 2')	
	#fig.suptitle('Bump position')
	plt.legend(handles=[p1,p2,p3,p4,p5,p6], ncol=6, loc='center', bbox_to_anchor=(0.5, 1.2)) #, prop=fontP

	fig.savefig("figs/trackingmean_%s_%d.png"%(SimulationName,sim), bbox_inches='tight')
	fig.savefig("figs/trackingmean_%s_%d.svg"%(SimulationName,sim), bbox_inches='tight')

	# fig, ax2 = plt.subplots(1,1,figsize=(3,3))
	# ax2.hist(mymaxb, density=True)
	# ax2.set_xlim(0,1)
	# ax2.set_xlabel('bump location in PPC')
	# ax2.set_ylabel('Pdf')
	# fig.savefig("figs/DistribAct_%s_%d.png"%(SimulationName,sim), bbox_inches='tight')