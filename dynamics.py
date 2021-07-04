#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 8 Feb

@author: vb
"""
import numpy as np
import random
import os
import time
import cProfile
import pstats
import sys
from numba import jit

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from pylab import rcParams

# the axes attributes need to be set before the call to subplot
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}, size=14)
rc('text', usetex=True)
rc('axes', edgecolor='black', linewidth=0.5)
rc('legend', frameon=False)
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # \boldmath

def main():
	########################################################################### BEGIN PARAMETERS ############################################################################

	#index=int(sys.argv[1])-1
	#params=np.loadtxt("params.txt")

	N=2000 #number of neurons

	tauhWM=0.01 # 0.1
	tauthetaWM=0.5 # 5

	tauhH=5. # 10
	tauthetaH=20. #20 

	DWM=0.0 # params[index,0] amount of adaptation of WM net
	DH=0. # params[index,1] amount of adaptation of H net

	beta=5. # activation sensitivity
	a=0.1 # sparsity 
	J0=0.4 # uniform inhibition
	eps=0.0 #params[index,0]

	AWMtoH=0.0 #np.linspace(0.1,1,10) #0.8 strength of connections from WM net to H net
	AHtoWM=0.25 #0.4 0.33 #np.linspace(0.1,1,10) #0.33 strength of connections from H net to WM net

	num_sims=1 # number of sessions
	num_trials=10 # number of trials within each session

	periodic = False # whether or not we want periodic boundary conditions
	
	stimulus_set = np.array([ [0.3,0.2], [0.4,0.3], [0.5,0.4], [0.6,0.5], [0.7,0.6], [0.8,0.7], \
								[0.2,0.3], [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.7,0.8], \
								[0.45, 0.5], [0.55, 0.5], [0.5, 0.45], [0.5, 0.55] ])

	deltax=0.05	

	deltat=0.4 # 400 ms
	delta_ISI=2 # in seconds

	t1val=1 # first stimulus given at 1 s
	t2val=t1val+deltat+delta_ISI # time at which second stimulus is given

	dt=0.001 # 0.01

	trialduration=1+deltat+delta_ISI+deltat+1.2 # seconds

	SimulationName="test/AHtoWM%.2f_tauhH%.2f_DWM%.2f_DH%.2f_eps%.2f_TISI%d"%(AHtoWM,tauhH,DWM,DH,eps,delta_ISI)
	
	SaveFullDynamics = 1

	# POINTS TO TAKE FOR READOUT
	if SaveFullDynamics == 1:
		tsave=np.arange(0,trialduration,dt) 
	else:
		tsave=np.array([t2val-deltat/2,t2val,t2val+deltat/2.])

	########################################################################### END PARAMETERS ############################################################################
	
	# NOW TRANSFORM 

	deltat=int(deltat/dt)
	delta_ISI=int(delta_ISI/dt)

	t1val=int(t1val/dt)
	t2val=int(t2val/dt)

	maxsteps=int(trialduration/dt)
	tsave=np.around(tsave/dt).astype(int)

	# define probabilities for equally spaced stimuli and also psychometric stimuli
	probas=np.zeros(len(stimulus_set)) 
	probas[:12]=0.9
	probas[12:]=0.1
	probas=probas/np.sum(probas)

	sim=0 #int(params[index,1])
	
	np.random.seed(sim) #int(params[index,2])) #time.time)	

	stimulus_set_new = stimulus_set #rescale(xmin,xmax,xmin_new,xmax_new,stimulus_set).round(decimals=3)

	#CREATE SIMULATION FOLDER
	if not os.path.exists(SimulationName):
		os.makedirs(SimulationName)

	if(SaveFullDynamics != 1):
		# this is for performance by stimulus scatter
		np.save("%s/stimulus_set.npy"%(SimulationName), stimulus_set_new)		

		# this is for psychometric curve
		np.save("%s/dstim_set.npy"%(SimulationName), np.unique(np.round_(stimulus_set_new[:,0]-stimulus_set_new[:,1], decimals=3)))		

	RingWM=MakeRing(N) # defines environment. 
	JWMtoWM=BuildJ(N,RingWM,J0=J0,a=a,periodic=periodic) # builds connectivity within WM net

	RingH=MakeRing(N)
	JHtoH=BuildJ(N,RingH,J0=J0,a=a,periodic=periodic) # builds connectivity within H net
	# no need to make inter network connectivity, as they are one-to-one    

	#for sim in range(num_sims):

	choices=np.random.choice(np.arange(len(stimulus_set_new)), num_trials, p=probas)
	stimuli=np.array([stimulus_set_new[i,:] for i in choices])

	VWM, VH, thetaWM, thetaH, hWM, hH, uWM, uH = np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)

	if SaveFullDynamics == 0:
		labels = np.zeros(num_trials)
		drift = np.zeros((num_trials,2))
		VWMsave = np.zeros((num_trials, len(tsave)*N))
		VHsave = np.zeros((num_trials, len(tsave)*N))

	for trial in range(num_trials):
		print(sim,trial)
		s=MakeStim(maxsteps,N,stimuli[trial,0],stimuli[trial,1],t1val,t2val,deltat=deltat,deltax=deltax)

		VWM_t,VH_t,thetaWM_t,thetaH_t, drift12, drift2end = UpdateNet(JWMtoWM, JHtoH, AWMtoH, AHtoWM, s,\
			VWM, VH, thetaWM, thetaH, hWM, hH, uWM, uH, tsave,\
			DWM=DWM, DH=DH, tauthetaWM=tauthetaWM, tauthetaH=tauthetaH, tauhWM=tauhWM, tauhH=tauhH, \
			maxsteps=maxsteps, dt=dt, beta=beta, N=N, x1=stimuli[trial,0], x2=stimuli[trial,1], \
			t1val=t1val, t2val=t2val, deltat=deltat, eps=eps)


		if(SaveFullDynamics == 1):
			PlotHeat(VWM_t,VH_t,thetaWM_t,thetaH_t,s,maxsteps,sim,trial,stimuli[trial,0],stimuli[trial,1],deltax,t1val,t2val,dt,N,SimulationName)
		else:	
			VWMsave[trial] = np.ravel(VWM_t)
			VHsave[trial] = np.ravel(VH_t)

			if stimuli[trial,0]>stimuli[trial,1]:
				labels[trial]=1

			# WHETHER DRIFT IS TOWARD MEAN or PREVIOUS STIMULUS
			if (trial > 0):
				drift[trial,0] = np.abs(drift12)*np.sign((np.mean(stimuli[:trial,:])-stimuli[trial,0])*drift12)				
				drift[trial,1] = np.abs(drift12)*np.sign((stimuli[trial-1,1]-stimuli[trial,0])*drift12)				

	#if(SaveFullDynamics == 1):
	#	continue 
	
	if(SaveFullDynamics == 0):
		# this is for history plot	
		np.save("%s/stimuli_sim%d"%(SimulationName, sim), stimuli)

		# these are for perceptron
		np.save("%s/label_sim%d"%(SimulationName, sim), labels)
		np.save("%s/VWM_sim%d"%(SimulationName, sim), VWMsave)
		np.save("%s/VH_sim%d"%(SimulationName, sim), VHsave)

		# drift
		np.save("%s/drift_sim%d"%(SimulationName, sim), drift)
	return

# FUNCTIONS

def rescale(xmin,xmax,xmin_new,xmax_new,stimulus_set):
	#stimulus_set_new=np.array((8,2))
	for i in range(np.shape(stimulus_set)[0]):
		for j in range(2):
			stimulus_set[i,j] = ((xmax_new - xmin_new)*(stimulus_set[i,j]-xmin))/(xmax-xmin) + xmin_new
	return stimulus_set


def MakeStim(maxsteps,N,x1,x2,t1,t2,deltat,deltax):
	s=np.zeros((maxsteps,N))
	s[t1:int(t1+deltat),int((x1-deltax)*N):int((x1+deltax)*N)]=1
	s[t2:int(t2+deltat),int((x2-deltax)*N):int((x2+deltax)*N)]=1
	return s

@jit(nopython=True)
def dot(x,y):
	return np.dot(x,y)

def UpdateNet(JWMtoWM, JHtoH, AWMtoH, AHtoWM, s, VWM, VH, thetaWM, thetaH, hWM, hH, uWM, uH, tsave, \
	DWM=0.1,DH=0.5,tauthetaWM=5.,tauthetaH=5.,tauhWM=0.1,tauhH=0.1,maxsteps=1000,dt=0.01,
	beta=1., N=1000, x1=0, x2=0, t1val=0, t2val=0, deltat=0, eps=0.):
	 
		VWMsave=np.zeros((len(tsave),N))
		VHsave=np.zeros((len(tsave),N))
		thetaWMsave=np.zeros((len(tsave),N))
		thetaHsave=np.zeros((len(tsave),N))

		k=0
		xafter1=0
		xbefore2=0
		xafter2=0
		xend=0
		#print(t1val+deltat+1,t2val-1,t2val+deltat+1,maxsteps)

		for step in range(maxsteps):
			thetaWM+=dt*(DWM*VWM-thetaWM)/tauthetaWM
			thetaH+=dt*(DH*VH-thetaH)/tauthetaH

			# UPDATE THE WM NET
			#hWM += dt*(np.dot(JWMtoWM,VWM) + VH*AHtoWM + s[step] - thetaWM - hWM)/tauhWM # + #random.uniform(0., 0.1)
			hWM += dt*(dot(JWMtoWM,VWM) + VH*AHtoWM + s[step] - thetaWM - hWM + random.uniform(-eps, eps))/tauhWM # + #random.uniform(0., 0.1)
			VWM = Logistic(hWM,beta)

			# UPDATE THE H NET
			#hH += dt*(np.dot(JHtoH,VH) + s[step] - thetaH - hH)/tauhH #+ random.uniform(0., 2.) + VWM*AWMtoH
			hH += dt*(dot(JHtoH,VH) + s[step] - thetaH - hH)/tauhH #+  + VWM*AWMtoH
			VH = Logistic(hH,beta)

			# TAKE SNAPSHOTS OF SYSTEM FOR READOUT
			if step in tsave:
				VWMsave[k,:]=VWM
				VHsave[k,:]=VH
				thetaWMsave[k,:]=thetaWM
				thetaHsave[k,:]=thetaH
				k+=1

			# COMPUTE BUMP DRIFT FROM END OF FIRST STIMULUS TO BEGINNING OF SECOND STIMULUS

			if (step==(t1val+deltat + 1)):
				xafter1+=np.argmax(VWM)/N
				
			if (step==(t2val - 1)):
				xbefore2+=np.argmax(VWM)/N

			# COMPUTE BUMP DRIFT FROM END OF SECOND STIMULUS TO END OF TRIAL

			if (step==(t2val+deltat + 1)):
				xafter2+=np.argmax(VWM)/N

			if (step==(maxsteps-1)):
				xend+=np.argmax(VWM)/N

		d12=xbefore2-xafter1	
		d2end=xend-xafter2
		#print(d12,d2end)
		#print("Dynamic step: "+str(step)+" done, mean: "+str(np.mean(V))+" sparsity: "+str(pow(np.mean(V),2)/np.mean(pow(V,2))))
		#print(Vsave)
		return VWMsave, VHsave, thetaWMsave, thetaHsave, d12, d2end

def PlotHeat(VWMs,VHs,thetaWMs,thetaHs,S,maxsteps,sim,trial,stim1,stim2,deltax,t1val,t2val,dt,N,SimulationName):
	fig, axs = plt.subplots(3, figsize=(9,4), num=1, clear=True)
	#Vs=np.load("1D/Vdynamics.npy")
	#S=np.load("1D/Stimuli.npy")
	print(np.shape(S))
	im = axs[0].imshow(S.T, cmap=cm.Greys, origin='lower', aspect='auto')#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
	axs[0].text(t1val+500,(stim1+deltax)*1000, '%.2f'%stim1)
	axs[0].text(t2val+500,(stim2+deltax)*1000, '%.2f'%stim2)
	im1 = axs[1].imshow(VWMs.T, cmap=cm.RdYlGn, origin='lower', aspect='auto', vmin=0, vmax=1)#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
	#im2 = axs[2].imshow(thetaWMs.T, cmap=cm.Blues, origin='lower', aspect='auto')#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
	im2 = axs[2].imshow(VHs.T, cmap=cm.RdYlGn, origin='lower', aspect='auto', vmin=0, vmax=1)#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
	#im4 = axs[4].imshow(thetaHs.T, cmap=cm.Blues, origin='lower', aspect='auto')#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
	
	#ax.axhline(y=400, color='k', linestyle='-')
	#ax.axhline(y=1200, color='k', linestyle='-')
	
	divider = make_axes_locatable(axs[0])
	cax = divider.append_axes("right", size="3%", pad=0.3)
	plt.colorbar(im,cax=cax,orientation='vertical')
	
	divider = make_axes_locatable(axs[1])
	cax = divider.append_axes("right", size="3%", pad=0.3)    
	plt.colorbar(im1,cax=cax,orientation='vertical')

	divider = make_axes_locatable(axs[2])
	cax = divider.append_axes("right", size="3%", pad=0.3)    
	plt.colorbar(im2,cax=cax,orientation='vertical')    
 
	# divider = make_axes_locatable(axs[3])
	# cax = divider.append_axes("right", size="3%", pad=0.3)    
	# plt.colorbar(im3,cax=cax,orientation='vertical')    
 
	# divider = make_axes_locatable(axs[4])
	# cax = divider.append_axes("right", size="3%", pad=0.3)    
	# plt.colorbar(im4,cax=cax,orientation='vertical')    
	
	# axs[4].set_ylabel("log(thetaH(x))")

	axs[0].set_xticks([])
	axs[1].set_xticks([])
	axs[2].set_xticks([])
	# axs[3].set_xticks([])
	axs[2].set_xticks(np.arange(0,maxsteps+1,500))
	axs[2].set_xticklabels(['%.2f'%(i*dt) for i in range(0,maxsteps+1,500)] )
	axs[2].set_xlabel("time [s]")

	axs[0].set_yticks([0,1000,2000])
	axs[1].set_yticks([0,1000,2000])
	axs[2].set_yticks([0,1000,2000])
	# axs[3].set_yticks(np.arange(0,N,200))
	# axs[4].set_yticks(np.arange(0,N,200))

	axs[0].set_yticklabels([0,0.5,1])
	axs[1].set_yticklabels([0,0.5,1])
	axs[2].set_yticklabels([0,0.5,1])
	# axs[3].set_yticklabels([i/N for i in range(0,N,200)])
	# axs[4].set_yticklabels([i/N for i in range(0,N,200)])

	#ax.set_yticklabels(['%.2f'%i for i in np.linspace(0,1,6)]);   
	#fig.tight_layout() 
	axs[0].set_ylabel("$I_{ext}$")    
	axs[1].set_ylabel("$V^{WM}$")
	axs[2].set_ylabel("$V^{PPC}$")
	# axs[3].set_ylabel("log(VH(x))")
	fig.savefig("%s/heatmap_sim%d_trial%d.png"%(SimulationName,sim,trial), bbox_inches='tight')
	#fig.close()

def K(x1,x2,N,J0=0.2,a=0.03,periodic=True,cutoff=None):
	d=x1-x2
	if periodic:
		if d>0.5:
			d=d-1.
		elif d<-0.5:
			d=d+1.
		return np.exp(-abs(d/a)) - J0
	else:
		if cutoff is None or abs(d) < cutoff:
			return np.exp(-abs(d/a)) - J0
		else:
			return 0

@jit(nopython=True)
def MakeRing(N):
		grid=np.zeros(N)
		for i in range(N):
			grid[i]=float(i)/float(N)
		return grid

def BuildJ(N,grid,a=0.03,J0=0.2,periodic=True):
	J=np.zeros((N,N))
	for i in range(N):
		for j in range(N):
			if i!=j:
				J[i][j]=K(grid[i],grid[j],N,J0=J0,a=a,periodic=periodic,cutoff=None) 
	return J/(N*a)

@jit(nopython=True)
def Logistic(h,beta):
	return 1./(1.+np.exp(-beta*h))        

def CorrAct(pos,bump_center=0.5):
	N=len(pos)
	V=np.zeros(N)
	for i in range(N):
		V[i]= K(pos[i],bump_center,N)
	#print(V)
	return V

if __name__ == "__main__":
	#cProfile.run('main()')
	main()
	# def f (a,x):
	#     return 1./(1.+np.exp(-a*x+3))

	# x = np.linspace(-5,10,100)
	# y = f(100,x)
	# plt.plot(x,y)
	# plt.show()
