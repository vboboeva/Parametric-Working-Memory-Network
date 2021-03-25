#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 8 Feb

@author: vb
"""
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

def main():

	SimulationName="testruns"
	N=1000 #number of neurons
	m=1 #number of maps

	tauhWM=0.1
	tauthetaWM=5.

	tauhH=3.
	tauthetaH=5.

	tauF=2
	U=0.3

	maxsteps=5000
	skipsteps=100
	dt=0.01

	beta=10. # activation sensitivity
	h0=0.  # static threshold
	a=0.03 # sparsity 
	J0=0.2 # uniform inhibition
	D=0.2 # amount of adaptation

	AWMtoH=0.8 #np.linspace(0.1,1,10) #0.8 strength of connections from WM net to H net
	AHtoWM=0.33 #np.linspace(0.1,1,10) #0.33 strength of connections from H net to WM net
	periodic = False # whether or not we want periodic boundary conditions
	stimulus_set= np.array([[0.6,0.68],[0.68,0.6],[0.68,0.76],[0.76,0.68],[0.76,0.84],[0.84,0.76],[0.84,0.92],[0.92,0.84], \
								[0.70,0.76],[0.72,0.76],[0.74,0.76],[0.75,0.76],[0.755,0.76],[0.765,0.76],[0.77,0.76],[0.78,0.76],[0.80,0.76],[0.82,0.76] ])

	print(np.shape(stimulus_set)[0])
	xmin=0.6
	xmax=0.92

	xmin_new=0.1
	xmax_new=0.92

	def rescale(xmin,xmax,xmin_new,xmax_new,stimulus_set):
		#stimulus_set_new=np.array((8,2))
		for i in range(np.shape(stimulus_set)[0]):
			for j in range(2):
				stimulus_set[i,j] = ((xmax_new - xmin_new)*(stimulus_set[i,j]-xmin))/(xmax-xmin) + xmin_new
		return stimulus_set

	stimulus_set_new = rescale(xmin,xmax,xmin_new,xmax_new,stimulus_set)

	# print(stimulus_set_new)
	# plt.scatter(stimulus_set_new[:,0],stimulus_set_new[:,1])
	# plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))
	# plt.show()

	deltat=int(2/dt)
	deltax=0.05	

	num_sims=1
	num_trials=200
	random.seed(1987)#time.time)

	#CREATE SIMULATION FOLDER
	if not os.path.exists(SimulationName):
		os.makedirs(SimulationName)

	RingWM=MakeRing(N,m) # defines environment. Ring is a m x N array, m number of maps, N number of units in the network
	JWMtoWM=BuildJ(N,RingWM,J0=J0,a=a,periodic=periodic) # builds connectivity within WM net

	RingH=MakeRing(N,m)
	JHtoH=BuildJ(N,RingH,J0=J0,a=a,periodic=periodic) # builds connectivity within H net
	# no need to make inter network connectivity, as they are one-to-one    

	t1val=int(1000)
	t2val=int(3000)

	for sim in range(num_sims):

		seq=np.arange(np.shape(stimulus_set)[0])
		vals=random.choices(seq, k=num_trials) #np.array([random.uniform(0.1, 0.9) for i in range(num_trials)])
		xvals=stimulus_set_new[vals]


		VWM, VH, thetaWM, thetaH, hWM, hH, uWM, uH = np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)

		stimuli = np.zeros((num_trials,2)) 
		labels = np.zeros(num_trials)
		VWMsave = np.zeros((num_trials, int(maxsteps/skipsteps)*N))
		print(np.shape(VWMsave))

		for trial in range(num_trials):
			print(trial)

			x1val=xvals[trial,0]
			x2val=xvals[trial,1]

			s=MakeStim(maxsteps,N,x1val,x2val,t1val,t2val,deltat=deltat,deltax=deltax)

			VWM_t,VH_t,thetaWM_t,thetaH_t = UpdateNet(JWMtoWM,JHtoH,AWMtoH,AHtoWM,s,\
				VWM, VH, thetaWM, thetaH, hWM, hH, uWM, uH, \
				D=D,tauthetaWM=tauthetaWM,tauthetaH=tauthetaH,tauhWM=tauhWM,tauhH=tauhH,tauF=tauF,U=U, \
				maxsteps=maxsteps,dt=dt,beta=beta,h0=h0,skipsteps=skipsteps,N=N)

			#PlotHeat(VWM_t,VH_t,thetaWM_t,thetaH_t,s,maxsteps,sim,trial)
			print(np.shape(VWM))
			print(np.shape(np.ravel(VWM)))
			VWMsave[trial] = np.ravel(VWM_t)
			stimuli[trial,0] = x1val
			stimuli[trial,1] = x2val

			#np.save("%s/s_sim%d_trial%d"%(SimulationName, sim, trial), s)
			if x1val>x2val:
				#print(trial)
				labels[trial]=1

		np.save("%s/stimuli_sim%d"%(SimulationName, sim), stimuli)
		np.save("%s/label_sim%d"%(SimulationName, sim), labels)
		np.save("%s/VWM_sim%d"%(SimulationName, sim), VWMsave)
		#np.save("%s/VH_sim%d_trial%d"%(SimulationName, sim, trial), VHsave)

	return

# FUNCTIONS

def MakeStim(maxsteps,N,x1,x2,t1,t2,deltat,deltax):
	s=np.zeros((maxsteps,N))
	s[t1:int(t1+deltat),int((x1-deltax)*N):int((x1+deltax)*N)]=1
	s[t2:int(t2+deltat),int((x2-deltax)*N):int((x2+deltax)*N)]=1
	return s

def UpdateNet(JWMtoWM, JHtoH, AWMtoH, AHtoWM, s, VWM, VH, thetaWM, thetaH, hWM, hH, uWM, uH, D=0.3,tauthetaWM=5.,tauthetaH=5.,tauhWM=0.1,tauhH=0.1,tauF=10.,U=0.1,maxsteps=1000,dt=0.01,beta=1.,h0=0.,skipsteps=10, N=1000):
	 
		#Vvec=np.zeros((maxsteps,N))
		VWMsave=np.zeros((int(maxsteps/skipsteps),N))
		VHsave=np.zeros((int(maxsteps/skipsteps),N))
		thetaWMsave=np.zeros((maxsteps,N))
		thetaHsave=np.zeros((maxsteps,N))

		k=0
		for step in range(maxsteps):
			thetaWM+=dt*(D*VWM-thetaWM)/tauthetaWM
			thetaH+=dt*(D*VH-thetaH)/tauthetaH

			# these are not being used for the moment
			uWM+=dt*(-(uWM-U)+tauF*U*VWM*(1.-uWM))            
			uH+=dt*(-(uH-U)+tauF*U*VH*(1.-uH))

			# UPDATE THE WM NET
			hWM += dt*(np.dot(JWMtoWM,VWM) + VH*AHtoWM + s[step] - thetaWM - hWM)/tauhWM #+ #random.uniform(0., 0.1)
			VWM = Logistic(hWM,beta,h0)

			# UPDATE THE H NET
			hH += dt*(np.dot(JHtoH,VH) + VWM*AWMtoH - thetaH - hH)/tauhH #+ random.uniform(0., 2.)
			VH = Logistic(hH,beta,h0)

			if(step%skipsteps==0):
				VWMsave[k,:]=VWM
				VHsave[k,:]=VH
				thetaWMsave[k,:]=thetaWM
				thetaHsave[k,:]=thetaH
				k+=1
			
			#print("Dynamic step: "+str(step)+" done, mean: "+str(np.mean(V))+" sparsity: "+str(pow(np.mean(V),2)/np.mean(pow(V,2))))
		#print(Vsave)
		return VWMsave, VHsave, thetaWMsave, thetaHsave

def PlotHeat(VWMs,VHs,thetaWMs,thetaHs,S,maxsteps,sim,trial):
	fig, axs = plt.subplots(5, figsize=(18,12) ) #
	#Vs=np.load("1D/Vdynamics.npy")
	#S=np.load("1D/Stimuli.npy")
	print(np.shape(S))
	im = axs[0].imshow(np.log(S.T), interpolation='bilinear', cmap=cm.Greys, origin='lower')#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
	im1 = axs[1].imshow(np.log(VWMs.T), interpolation='bilinear', cmap=cm.RdYlGn, origin='lower')#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
	im2 = axs[2].imshow(thetaWMs.T, interpolation='bilinear', cmap=cm.Blues, origin='lower')#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
	im3 = axs[3].imshow(np.log(VHs.T), interpolation='bilinear', cmap=cm.RdYlGn, origin='lower')#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
	im4 = axs[4].imshow(thetaHs.T, interpolation='bilinear', cmap=cm.Blues, origin='lower')#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
	
	#ax.axhline(y=400, color='k', linestyle='-')
	#ax.axhline(y=1200, color='k', linestyle='-')
	divider = make_axes_locatable(axs[0])
	cax = divider.append_axes("top", size="10%", pad=0.3)
	plt.colorbar(im,cax=cax,orientation='horizontal')
	
	divider = make_axes_locatable(axs[1])
	cax = divider.append_axes("top", size="10%", pad=0.3)    
	plt.colorbar(im1,cax=cax,orientation='horizontal')

	divider = make_axes_locatable(axs[2])
	cax = divider.append_axes("top", size="10%", pad=0.3)    
	plt.colorbar(im2,cax=cax,orientation='horizontal')    
 
	divider = make_axes_locatable(axs[3])
	cax = divider.append_axes("top", size="10%", pad=0.3)    
	plt.colorbar(im3,cax=cax,orientation='horizontal')    
 
	divider = make_axes_locatable(axs[4])
	cax = divider.append_axes("top", size="10%", pad=0.3)    
	plt.colorbar(im4,cax=cax,orientation='horizontal')    
	
	# divider = make_axes_locatable(axs[2])
	# cax = divider.append_axes("top", size="50%", pad=0.3)    
	# plt.colorbar(im2,cax=cax,orientation='horizontal')    
	# ax.set_xticks(np.arange(0,1200,200))
	# ax.set_xticklabels(np.arange(0,1200,200))
	# ax.set_yticks(np.arange(0,1200,200))

	axs[0].set_ylabel("log(S(x))", fontsize=14)    
	axs[1].set_ylabel("log(VWM(x))", fontsize=14)
	axs[2].set_ylabel("log(thetaWM(x))", fontsize=14)
	axs[3].set_ylabel("log(VH(x))", fontsize=14)
	axs[4].set_ylabel("log(thetaH(x))", fontsize=14)
	axs[1].set_xlabel("time", fontsize=14)
	# ax.set_yticklabels(['%.2f'%i for i in np.linspace(0,1,6)]);   
	fig.tight_layout() 
	fig.savefig("heatmap_sim%d_trial%d.png"%(sim,trial))

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


def MakeRing(N,m):
		grid=np.zeros((m,N))
		tempgrid=np.zeros(N)
		for i in range(N):
			tempgrid[i]=float(i)/float(N)
		grid[0][:]=tempgrid    
		for j in range(1,m):
			random.shuffle(tempgrid)
			grid[j][:]=tempgrid
		return grid

def BuildJ(N,grid,a=0.03,J0=0.2,periodic=True):
	J=np.zeros((N,N))
	for i in range(N):
		for j in range(N):
			for k in range(len(grid)):
				x1=grid[k][i]
				x2=grid[k][j]
				if i!=j:
					J[i][j]=K(x1,x2,N,J0=J0,a=a,periodic=periodic,cutoff=None) 
	return J/(N*a)

def Logistic(h,beta,h0):
	return 1./(1.+np.exp(-beta*(h-h0)))        

def CorrAct(pos,bump_center=0.5):
	N=len(pos)
	V=np.zeros(N)
	for i in range(N):
		V[i]= K(pos[i],bump_center,N)
	#print(V)
	return V

if __name__ == "__main__":
	main()

	# def f (a,x):
	#     return 1./(1.+np.exp(-a*x+3))

	# x = np.linspace(-5,10,100)
	# y = f(100,x)
	# plt.plot(x,y)
	# plt.show()