#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 8 Feb

@author: vb
"""
import numpy as np
import random
import os
#import matplotlib.pyplot as plt
#from matplotlib import cm
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

def main():

    SimulationName="data"
    N=1000 #number of neurons
    m=1 #number of maps

    tauhWM=0.1
    tauthetaWM=5.

    tauhH=3.
    tauthetaH=5.

    tauF=2
    U=0.3

    maxsteps=5000
    skipsteps=1
    dt=0.01

    beta=10. # activation sensitivity
    h0=0.  # static threshold
    J0=0.2 # uniform inhibition
    a=0.03 # sparsity 
    D=0.3 # amount of adaptation
    AWMtoHvals=[0.1]#np.linspace(0.1,1,10) #0.8 strength of connections from WM net to H net
    AHtoWMvals=[0.1]#np.linspace(0.1,1,10) #0.33 strength of connections from H net to WM net
    periodic = False # whether or not we want periodic boundary conditions

    num_sims=1
    random.seed(1987)#time.time)

    x1vals=np.array([random.uniform(0.1, 0.9) for i in range(num_sims)])
    x2vals=np.array([random.uniform(0.1, 0.9) for i in range(num_sims)])

    t1=int(0/dt)
    t2=int(5/dt)
    deltat=int(2/dt)
    deltax=0.05

    #CREATE SIMULATION FOLDER
    if not os.path.exists(SimulationName):
        os.makedirs(SimulationName)

    # labels=np.zeros(len(x1vals))
    # for i in range(len(x1vals)):
    #     if (x1vals[i]-x2vals[i] >= 0. ):
    #         labels[i]=1
    #     else:
    #         labels[i]=0

    # np.save(SimulationName+"/labels",labels)

    RingWM=MakeRing(N,m) # defines environment. Ring is a m x N array, m number of maps, N number of units in the network
    JWMtoWM=BuildJ(N,RingWM,J0=J0,a=a,periodic=periodic) # builds connectivity within WM net

    #np.save(SimulationName+"/pfc",Ring1)

    RingH=MakeRing(N,m)
    JHtoH=BuildJ(N,RingH,J0=J0,a=a,periodic=periodic) # builds connectivity within H net

    # no need to make within network connectivity, as they are one-to-one    

    for i in range(len(x1vals)):
        x1=x1vals[i]
        x2=x2vals[i]
        print(i,x1,x2)
        
        s=MakeStim(maxsteps,N,t1=t1,t2=t2,x1=x1,x2=x2,deltat=deltat,deltax=deltax)

        for AWMtoH in AWMtoHvals:
        	for AHtoWM in AHtoWMvals:
	        	np.save("%s/s_x1_%.2f_x2_%.2f_t1_%d_t2_%d_AWMtoH%.2f_AHtoWM%.2f"%(SimulationName,x1,x2,t1,t2,AWMtoH,AHtoWM), s)
		        VWMsave, VHsave = UpdateNet(JWMtoWM,JHtoH,AWMtoH,AHtoWM,s,D=D,tauthetaWM=tauthetaWM,tauthetaH=tauthetaH,tauhWM=tauhWM,tauhH=tauhH,tauF=tauF,U=U,maxsteps=maxsteps,dt=dt,beta=beta,h0=h0,skipsteps=skipsteps,N=N)
	        	np.save("%s/VWM_x1_%.2f_x2_%.2f_t1_%d_t2_%d_AWMtoH%.2f_AHtoWM%.2f"%(SimulationName,x1,x2,t1,t2,AWMtoH,AHtoWM), VWMsave)
	        	np.save("%s/VH_x1_%.2f_x2_%.2f_t1_%d_t2_%d_AWMtoH%.2f_AHtoWM%.2f"%(SimulationName,x1,x2,t1,t2,AWMtoH,AHtoWM), VHsave)
	        	#print(np.shape(data))
    #print("Dynamics terminated, result saved")
    return

# FUNCTIONS

def UpdateNet(JWMtoWM,JHtoH,AWMtoH,AHtoWM,s,D=0.3,tauthetaWM=5.,tauthetaH=5.,tauhWM=0.1,tauhH=0.1,tauF=10.,U=0.1,maxsteps=1000,dt=0.01,beta=1.,h0=0.,skipsteps=10, N=1000):
     
        #Vvec=np.zeros((maxsteps,N))
        VWMsave=np.zeros((int(maxsteps/skipsteps),N))
        VHsave=np.zeros((int(maxsteps/skipsteps),N))
        #thetavec=np.zeros((maxsteps,N))
        #uvec=np.zeros((maxsteps,N))
        VWM, VH, thetaWM, thetaH, hWM, hH, uWM, uH = np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N),np.zeros(N)

        k=0
        for step in range(maxsteps):
            thetaWM+=dt*(D*VWM-thetaWM)/tauthetaWM
            thetaH+=dt*(D*VH-thetaH)/tauthetaH

            uWM+=dt*(-(uWM-U)+tauF*U*VWM*(1.-uWM))            
            uH+=dt*(-(uH-U)+tauF*U*VH*(1.-uH))

            # UPDATE THE WM NET
            hWM += dt*(np.dot(JWMtoWM,VWM) + VH*AHtoWM + s[step] - thetaWM - hWM)/tauhWM #+ #random.uniform(0., 0.1)
            VWM = Logistic(hWM,beta,h0)

            # UPDATE THE H NET
            hH += dt*(np.dot(JHtoH,VH) + VWM*AWMtoH - thetaH - hH)/tauhH #+ random.uniform(0., 2.)
            VH = Logistic(hH,beta,h0)

            #uvec[step,:]=u
            #thetavec[step,:]=theta
            #Vvec[step,:]=V

            if(step%skipsteps==0):
                VWMsave[k,:]=VWM
                VHsave[k,:]=VH
                k+=1
            
            #print("Dynamic step: "+str(step)+" done, mean: "+str(np.mean(V))+" sparsity: "+str(pow(np.mean(V),2)/np.mean(pow(V,2))))
        #print(Vsave)
        return VWMsave, VHsave

def PlotHeat(VWMs,VHs,S,maxsteps,x1,x2,t1,t2,AWMtoH,AHtoWM):
    print(x1,x2)
    fig, axs = plt.subplots(3, figsize=(9,6)) 
    #Vs=np.load("1D/Vdynamics.npy")
    #S=np.load("1D/Stimuli.npy")
    print(np.shape(S))
    im = axs[0].imshow(np.log(VWMs.T), interpolation='bilinear', cmap=cm.RdYlGn, origin='lower')#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
    im1 = axs[1].imshow(np.log(VHs.T), interpolation='bilinear', cmap=cm.RdYlGn, origin='lower')#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
    im2 = axs[2].imshow(np.log(S.T), interpolation='bilinear', cmap=cm.Greys, origin='lower')#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
    #im2 = axs[2].imshow(np.log(us.T), interpolation='bilinear', cmap=cm.Blues, origin='lower')#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
    
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
    
    # divider = make_axes_locatable(axs[2])
    # cax = divider.append_axes("top", size="50%", pad=0.3)    
    # plt.colorbar(im2,cax=cax,orientation='horizontal')    
    # ax.set_xticks(np.arange(0,1200,200))
    # ax.set_xticklabels(np.arange(0,1200,200))
    # ax.set_yticks(np.arange(0,1200,200))

    axs[0].set_ylabel("log(VWM(x))", fontsize=14)
    axs[1].set_ylabel("log(VH(x))", fontsize=14)
    axs[2].set_ylabel("log(S(x))", fontsize=14)    
    axs[1].set_xlabel("time", fontsize=14)
    # ax.set_yticklabels(['%.2f'%i for i in np.linspace(0,1,6)]);   
    fig.tight_layout() 
    fig.savefig("heatmap_x1_%.2f_x2_%.2f_t1_%d_t2_%d_AWMtoH%.2f_AHtoWM%.2f.png"%(x1,x2,t1,t2,AWMtoH,AHtoWM))

def MakeStim(maxsteps,N,t1,t2,x1,x2,deltat,deltax):
    s=np.zeros((maxsteps,N))
    s[t1:int(t1+deltat),int((x1-deltax)*N):int((x1+deltax)*N)]=1
    s[t2:int(t2+deltat),int((x2-deltax)*N):int((x2+deltax)*N)]=1
    return s


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
                    J[i][j]=K(x1,x2,N,a=a,J0=J0,periodic=periodic) 
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