import numpy as np
import random
import os
import matplotlib.pyplot as plt
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
import color_palette as cp
from helper_mymax import mymax

def make_data(SimulationName, N, num_sims, psycholabel):
	stimulus_set=np.load("data/" + SimulationName+"/stimulus_set.npy")
	delay=np.empty((0,1), float)
	# drift=np.empty((0,2), float)
	labels=np.empty((0,1), float)
	stimuli=np.empty((0,2), float)
	readout=np.empty((0,1), float)
	VHmax=np.empty((0,1), float)

	for sim in range(num_sims):
		myfile="data/" + SimulationName+'/inf_sim%d.npy'%sim
		if os.path.isfile(myfile):
			delay=np.append(delay, np.load("data/" + SimulationName+"/inf_sim%d.npy"%sim)[0,:])
			# drift=np.append(drift, np.transpose(np.load("data/" + SimulationName+"/inf_sim%d.npy"%sim)[1:3,:]), axis=0)
			labels=np.append(labels, np.load("data/" + SimulationName+"/inf_sim%d.npy"%sim)[3,:])
			stimuli=np.append(stimuli, np.transpose(np.load("data/" + SimulationName+"/inf_sim%d.npy"%sim)[4:,:]), axis=0)
			x1=np.argmax(np.load("data/" + SimulationName+"/VWM_sim%d.npy"%sim)[:,np.arange(0,N)], axis=1)
			x2=np.argmax(np.load("data/" + SimulationName+"/VWM_sim%d.npy"%sim)[:,np.arange(2*N,3*N)], axis=1)
			x=mymax(x1, x2)
			readout=np.append(readout, x)
			
			VH=np.load("data/"+ SimulationName+"/VH_sim%d.npy"%sim)[:,0:N]
			VHmax=np.append(VHmax, np.argmax(VH,axis=1)/N)
	if psycholabel == False:
		labels=1.-labels
		readout=1.-readout

	return stimulus_set, stimuli, labels, readout, delay, VHmax	
