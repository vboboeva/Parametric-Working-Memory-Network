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
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from pylab import rcParams

# the axes attributes need to be set before the call to subplot
rc('font',**{'family':'sans-serif','sans-serif':['Arial']}, size=10)
rc('text', usetex=True)
rc('axes', edgecolor='black', linewidth=0.5)
rc('legend', frameon=False)
rcParams['ytick.direction'] = 'in'
rcParams['xtick.direction'] = 'in'
rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # \boldmath

def main(tauhH=0.5, tauthetaH=7.5, DH=0.3, JeHtoWM=0.5, ITI=5, sim=0):

	########################################################################### BEGIN PARAMETERS ############################################################################
	N=2000 # number of neurons

	tauhWM=0.01 # neuronal timescale in WM
	tauthetaWM=0.5 # adaptation timescale in WM (not used in default net params)

	# `tauhH` neuronal timescale in PPC
	# `tauthetaH` adaptation timescale in PPC

	DWM=0.0 # amount of adaptation of WM net
	# `DH` amount of adaptation of PPC

	beta=5. # activation sensitivity
	a=0.02 # sparsity 
	JeWM=1.
	JeH=1.
	# `JeHtoWM` strength of connections from PPC to WM net
	J0=0.2 # uniform inhibition
	eps=0.0 # amplitude of noise
	# `ITI` inter-trial interval
	# `sim` simulation number

	num_sims=1 # number of sessions
	num_trials=20 # number of trials within each session
	periodic = False # whether or not we want periodic boundary conditions
	
	stimulus_set = np.array([ [0.3,0.2], [0.4,0.3], [0.5,0.4], [0.6,0.5], [0.7,0.6], [0.8,0.7], \
								[0.2,0.3], [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.7,0.8], \
								[0.45, 0.5], [0.55, 0.5], [0.5, 0.45], [0.5, 0.55] ])

	delta_ISI=np.array([2,6,10]) # the durations of the delay interval

	dt=0.001 
	
	SimulationName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_eps%.2f_ITI%.1f"%(JeHtoWM,tauhH,tauthetaH,DH,eps,ITI)
	SaveFullDynamics = 1
	Spread_connectivity = 0

	# define probabilities for equally spaced stimuli and also psychometric stimuli
	probas=np.zeros(len(stimulus_set)) 
	probas[:12]=0.9
	probas[12:]=0.1
	probas=probas/np.sum(probas)
	deltax=0.05	

	np.random.seed(sim) #int(params[2])) #time.time)	

	stimulus_set_new = stimulus_set #rescale(xmin,xmax,xmin_new,xmax_new,stimulus_set).round(decimals=3)

	#CREATE SIMULATION FOLDER
	if not os.path.exists(SimulationName):
		os.makedirs(SimulationName)

	if(SaveFullDynamics != 1):
		# this is for performance by stimulus scatter
		np.save("%s/stimulus_set.npy"%(SimulationName), stimulus_set_new)		

		# this is for psychometric curve
		np.save("%s/dstim_set.npy"%(SimulationName), np.unique(np.round_(stimulus_set_new[:,0]-stimulus_set_new[:,1], decimals=3)))		

	Ring=MakeRing(N) # defines environment. 

	JWMtoWM=BuildJ(N,Ring,Je=JeWM,J0=J0,a=a,periodic=periodic, selfcon=0) # builds connectivity within WM net

	JHtoH=BuildJ(N,Ring,Je=JeH,J0=J0,a=a,periodic=periodic, selfcon=0) # builds connectivity within H net

	if Spread_connectivity == 0:
		JHtoWM=JeHtoWM*np.identity(N) 
	elif Spread_connectivity == 1:
		JHtoWM=BuildJ(N,Ring,Je=JeHtoWM,J0=0,a=0.0005,periodic=periodic, selfcon=1)

	choices=np.random.choice(np.arange(len(stimulus_set_new)), num_trials, p=probas)
	stimuli=np.array([stimulus_set_new[i,:] for i in choices])

	choices1=np.random.choice(np.arange(len(delta_ISI)), num_trials, p=np.ones(len(delta_ISI))*(1./len(delta_ISI)))
	delay_intervals=np.array([delta_ISI[i] for i in choices1])

	VWM, VH, thetaWM, thetaH, hWM, hH, uWM, uH = np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)

	if SaveFullDynamics == 0:
		labels = np.zeros(num_trials)
		disp = np.zeros((num_trials,2))
		VWMsave = np.zeros((num_trials, 3*N))
		VHsave = np.zeros((num_trials, 3*N))			

	for trial in range(num_trials):
		delay=delay_intervals[trial]
		deltat=0.4 # 400 ms
		t1val=1 # first stimulus given at 1 s
		t2val=t1val+deltat+delay # time at which second stimulus is given
		trialduration=1.+deltat+delay+deltat+ITI # seconds

		print(trial, stimuli[trial,0], stimuli[trial,1], delay)

		# POINTS TO TAKE FOR READOUT
		if SaveFullDynamics == 1:
			tsave=np.arange(0,trialduration,dt)
		else:
			tsave=np.array([t2val-deltat/2,t2val,t2val+deltat/2.])

		# NOW TRANSFORM 
		delay=int(delay/dt)
		t1val=int(t1val/dt)
		t2val=int(t2val/dt)
		deltat=int(deltat/dt)

		maxsteps=int(trialduration/dt)
		tsave=np.around(tsave/dt).astype(int)

		# MAKE THE STIMULI
		s=MakeStim(maxsteps,N,stimuli[trial,0],stimuli[trial,1],t1val,t2val,deltat=deltat,deltax=deltax)

		# RUN DYNAMICS
		VWM_t,VH_t,thetaWM_t,thetaH_t, disp12, disp2end = UpdateNet(JWMtoWM, JHtoH, JHtoWM, s, \
			VWM, VH, thetaWM, thetaH, hWM, hH, uWM, uH, tsave,\
			DWM=DWM, DH=DH, tauthetaWM=tauthetaWM, tauthetaH=tauthetaH, tauhWM=tauhWM, tauhH=tauhH, \
			maxsteps=maxsteps, dt=dt, beta=beta, N=N, x1=stimuli[trial,0], x2=stimuli[trial,1], \
			t1val=t1val, t2val=t2val, deltat=deltat, eps=eps)

		# TO VISUALIZE THE DYNAMICS FOR A FEW TRIALS 
		if(SaveFullDynamics == 1):
			PlotHeat(VWM_t,VH_t,thetaWM_t,thetaH_t,s,maxsteps,sim,trial,stimuli[trial,0],stimuli[trial,1],deltax,t1val,t2val,dt,N,SimulationName)

		# COLLECT BEHAVIOR OF NETWORK FOR MANY TRIALS
		else:	
			VWMsave[trial] = np.ravel(VWM_t)
			VHsave[trial] = np.ravel(VH_t)

			if stimuli[trial,0] > stimuli[trial,1]:
				labels[trial]=1

			# WHETHER DISPLACEMENT IS TOWARD MEAN or PREVIOUS STIMULUS
			if (trial > 0):
				disp[trial,0] = np.abs(disp12)*np.sign((np.mean(stimuli[:trial,:])-stimuli[trial,0])*disp12)				
				disp[trial,1] = np.abs(disp12)*np.sign((stimuli[trial-1,1]-stimuli[trial,0])*disp12)				

	# SAVE DATA WHEN COLLECTING BEHAVIOR FOR MANY TRIALS
	if(SaveFullDynamics == 0):
		A = np.vstack((delay_intervals,disp[:,0],disp[:,1],labels,stimuli[:,0],stimuli[:,1]))
		np.save("%s/inf_sim%d"%(SimulationName, sim), A)
		np.save("%s/VWM_sim%d"%(SimulationName, sim), VWMsave)
		np.save("%s/VH_sim%d"%(SimulationName, sim), VHsave)
	return

# FUNCTIONS

def rescale(xmin,xmax,xmin_new,xmax_new,stimulus_set):
	for i in range(np.shape(stimulus_set)[0]):
		for j in range(2):
			stimulus_set[i,j] = ((xmax_new - xmin_new)*(stimulus_set[i,j]-xmin))/(xmax-xmin) + xmin_new
	return stimulus_set

def MakeStim(maxsteps,N,x1,x2,t1,t2,deltat,deltax):
	s=np.zeros((maxsteps,N))
	s[t1:int(t1+deltat),int((x1-deltax)*N):int((x1+deltax)*N)]=1
	s[t2:int(t2+deltat),int((x2-deltax)*N):int((x2+deltax)*N)]=1
	return s

def UpdateNet(JWMtoWM, JHtoH, JHtoWM, s, VWM, VH, thetaWM, thetaH, hWM, hH, uWM, uH, tsave, \
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
		for step in range(maxsteps):
			# thetaWM+=dt*(DWM*VWM-thetaWM)/tauthetaWM
			thetaH+=dt*(DH*VH-thetaH)/tauthetaH

			# UPDATE THE WM NET
			hWM += dt*(np.dot(JWMtoWM,VWM) + np.dot(JHtoWM,VH) + s[step] - hWM + random.uniform(-eps, eps))/tauhWM  # - thetaWM if we want adaptation in WM net too
			VWM = Logistic(hWM,beta)

			# UPDATE THE H NET
			hH += dt*(np.dot(JHtoH,VH) + s[step] - thetaH - hH)/tauhH
			VH = Logistic(hH,beta)

			# TAKE SNAPSHOTS OF SYSTEM FOR READOUT
			if step in tsave:
				VWMsave[k,:]=VWM
				VHsave[k,:]=VH
				thetaWMsave[k,:]=thetaWM
				thetaHsave[k,:]=thetaH
				k+=1

			# COMPUTE BUMP DISPLACEMENT FROM END OF FIRST STIMULUS TO BEGINNING OF SECOND STIMULUS
			if (step==(t1val+deltat + 1)):
				xafter1+=np.argmax(VWM)/N
				
			if (step==(t2val - 1)):
				xbefore2+=np.argmax(VWM)/N

			# COMPUTE BUMP DISPLACEMENT FROM END OF SECOND STIMULUS TO END OF TRIAL
			if (step==(t2val+deltat + 1)):
				xafter2+=np.argmax(VWM)/N

			if (step==(maxsteps-1)):
				xend+=np.argmax(VWM)/N

		d12=xbefore2-xafter1	
		d2end=xend-xafter2
		return VWMsave, VHsave, thetaWMsave, thetaHsave, d12, d2end

# PLOT HEATMAP OF NET DYNAMICS (WORKS ONLY WHEN SaveFullDynamics = 1)
def PlotHeat(VWMs,VHs,thetaWMs,thetaHs,S,maxsteps,sim,trial,stim1,stim2,deltax,t1val,t2val,dt,N,SimulationName):
	
	fig, axs = plt.subplots(3, figsize=(4,2.3), num=1, clear=True)
	im = axs[0].imshow(S.T, cmap=cm.Greys, origin='lower', aspect='auto')
	axs[0].text(t1val+500,(stim1+deltax)*1000, '%.2f'%stim1)
	axs[0].text(t2val+500,(stim2+deltax)*1000, '%.2f'%stim2)
	
	im1 = axs[1].imshow(VWMs.T, cmap=cm.Greens, origin='lower', aspect='auto', vmin=0, vmax=1)
	im2 = axs[2].imshow(VHs.T, cmap=cm.Greens, origin='lower', aspect='auto', vmin=0, vmax=1)	
	divider = make_axes_locatable(axs[0])
	cax = divider.append_axes("right", size="3%", pad=0.3)
	plt.colorbar(im,cax=cax,orientation='vertical')
	
	divider = make_axes_locatable(axs[1])
	cax = divider.append_axes("right", size="3%", pad=0.3)    
	plt.colorbar(im1,cax=cax,orientation='vertical')

	divider = make_axes_locatable(axs[2])
	cax = divider.append_axes("right", size="3%", pad=0.3)    
	plt.colorbar(im2,cax=cax,orientation='vertical')    
 
	axs[0].set_xticks([])
	axs[0].set_yticks([0,1000,2000])
	axs[0].set_yticklabels([0,0.5,1])
	axs[0].set_ylabel("Neuron position [a.u.]")    

	axs[1].set_xticks([])
	axs[1].set_yticks([])
	axs[1].set_yticklabels([])
	
	axs[2].set_xticks(np.arange(0,maxsteps+1,1000))
	axs[2].set_xticklabels(['%d'%(i*dt) for i in range(0,maxsteps+1,1000)] )
	axs[2].set_yticks([])
	axs[2].set_yticklabels([])
	axs[2].set_xlabel("Time [s]")

	fig.savefig("%s/dynamics_sim%d_trial%d.png"%(SimulationName,sim,trial), bbox_inches='tight')
	fig.savefig("%s/dynamics_sim%d_trial%d.svg"%(SimulationName,sim,trial), bbox_inches='tight')

# CONNECTIVITY KERNEL
def K(x1,x2,N,Je=1.,J0=0.2,a=0.03,periodic=True,cutoff=None):
	d=x1-x2
	if periodic:
		if d>0.5:
			d=d-1.
		elif d<-0.5:
			d=d+1.
		return Je*np.exp(-abs(d/a)) - J0
	else:
		if cutoff is None or abs(d) < cutoff:
			return Je*np.exp(-abs(d/a)) - J0
		else:
			return 0

def MakeRing(N):
		grid=np.zeros(N)
		for i in range(N):
			grid[i]=float(i)/float(N)
		return grid

def BuildJ(N,grid,a=0.03,Je=1,J0=0.2,periodic=True, selfcon=0):
	J=np.zeros((N,N))
	for i in range(N):
		for j in range(N):
			if selfcon == 1:
				J[i][j]=K(grid[i],grid[j],N,Je=Je,J0=J0,a=a,periodic=periodic,cutoff=None) 
			else:
				if i!=j:
					J[i][j]=K(grid[i],grid[j],N,Je=Je,J0=J0,a=a,periodic=periodic,cutoff=None) 
	return J/(N*a)

def Logistic(h,beta):
	return 1./(1.+np.exp(-beta*h))        

def CorrAct(pos,bump_center=0.5):
	N=len(pos)
	V=np.zeros(N)
	for i in range(N):
		V[i]= K(pos[i],bump_center,N)
	return V

if __name__ == "__main__":
	
	tauhH=float(sys.argv[1])
	tauthetaH=float(sys.argv[2])
	DH=float(sys.argv[3])
	JeHtoWM=float(sys.argv[4])
	ITI=float(sys.argv[5])
	sim=int(sys.argv[6])

	main(tauhH, tauthetaH, DH, JeHtoWM, ITI, sim)