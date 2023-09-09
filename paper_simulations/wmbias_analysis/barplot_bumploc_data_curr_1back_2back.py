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

num_stimpairs=12
num_trials=1000
trialback=1
vicinity=0.05
bump_thresh=0.4

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
			print(myfile)
			stimuli=np.append(stimuli, np.transpose(np.load("data/" + SimulationName+"/inf_sim%d.npy"%sim)[4:,:]), axis=0)
			VWM=np.append(VWM, np.load( 'data/'+SimulationName+"/VWM_sim%d.npy"%sim)[:,takevalsWM], axis=0)
			VH=np.append(VH, np.load( 'data/'+SimulationName+"/VH_sim%d.npy"%sim)[:,takevalsH], axis=0)

	return stimuli, VWM, VH	


SimulationName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_eps%.2f_ITI%.1f"%(AHtoWM, tauhH, tauthetaH, DH, eps, ITI)

stimuli, VWM, VH = make_data()

fig, axs = plt.subplots(1,1,figsize=(2.5,1.25))#, num=1, clear=True)

VV=[VH, VWM]
ind = np.arange(8)
width = 0.3       
colors=[cp.pink, cp.violet2]
labels=['PPC', 'WM']

p1=[]
p2=[]
p3=[]
for i in range(2):
	V=VV[i]
	# find which trials are those in which the bump corresponds to the stimulus
	count=np.zeros(8)
	tot=0
	bumpmaxes=[]
	for trial in range(len(stimuli)):
		if trial in np.arange(0,num_trials*num_sims,500) or trial in np.arange(1,num_trials*num_sims,500): # remove the first two trials of every simulation since no hist effects:
			continue	
		else:
			posmax=np.argmax(V[trial,:])#/N
			# print(V[trial,posmax])
			bumpmaxes+=[V[trial,posmax]]
			if V[trial,posmax] > bump_thresh: # threshold for bump detection, according to distribution of bump height
				posmax=posmax/N
				meanstim=np.mean(stimuli[:trial,:])
				aboolean=False
				if np.isclose(stimuli[trial,0], posmax, atol=vicinity):
					count[7]+=1
					aboolean=True
				if np.isclose(stimuli[trial-1,1], posmax, atol=vicinity):
					count[6]+=1
					aboolean=True				
				if np.isclose(stimuli[trial-1,0], posmax, atol=vicinity):
					count[5]+=1
					aboolean=True				
				if np.isclose(stimuli[trial-2,1], posmax, atol=vicinity):
					count[4]+=1
					aboolean=True				
				if np.isclose(stimuli[trial-2,0], posmax, atol=vicinity):
					count[3]+=1
					aboolean=True				
				if np.isclose(meanstim, posmax, atol=vicinity):
					count[2]+=1
					aboolean=True				
				if aboolean==False:
					count[1]+=1
			else:
				count[0]+=1
			tot+=1
	# plt.hist(bumpmaxes)	# plt.show()
	print('dissipation',count[0]/tot)
	print('displacement',count[1]/tot)
	np.save("bumpmaxes_%s"%(labels[i]),bumpmaxes)
	if(i == 0):
		p1=axs.bar(ind+i*width*np.ones(len(ind)), count/tot, width=0.3, color=colors[i], label=labels[i])
	if (i == 1):
		p2=axs.bar(ind+i*width*np.ones(len(ind)), count/tot, width=0.3, color=colors[i], label=labels[i])

p3=axs.axhline(0.167, ls='--', color='black', label='Chance probability')

axs.set_xticks(ind + width / 2)
axs.set_xticklabels(['diss','disp','$\\langle s \\rangle$','$s_1^{t-2}$','$s_2^{t-2}$','$s_1^{t-1}$','$s_2^{t-1}$', '$s_1^{t}$'])
axs.set_ylabel("Fraction")
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
#fig.legend(loc='upper center')
fig.legend(handles=[p1,p2,p3], bbox_to_anchor=(0.55, 1.1), loc='upper center') #, prop=fontP
fig.savefig("figs/barplot_%s.png"%(SimulationName), bbox_inches='tight')
fig.savefig("figs/barplot_%s.svg"%(SimulationName), bbox_inches='tight')

fig, axs = plt.subplots(1,2,figsize=(3,1.5))#, num=1, clear=True)
A=np.load('bumpmaxes_PPC.npy')
print(np.min(A), np.max(A))
axs[0].hist(A, color=cp.pink, density=True, label='PPC')
axs[0].set_xlim(0,1.1)
B=np.load('bumpmaxes_WM.npy')
axs[1].hist(B, color=cp.violet2, density=True, label='WM')
axs[1].set_xlim(0.9,1)
fig.legend()
fig.savefig("figs/bumpmax_distrib_%s.svg"%(SimulationName), bbox_inches='tight')
