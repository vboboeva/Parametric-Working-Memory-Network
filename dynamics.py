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


def main():

    SimulationName="1D"
    N=1000 #number of neurons
    m=1 #number of maps
    gamma=10. 
    tauh=0.1
    tautheta=10.
    maxsteps=10000
    skipsteps=100
    dt=0.01
    beta=10.
    h0=0.
    J0=0.2
    c=0.04 # sparsity of the non-zero connections 
    sigma=20
    x1=0.3
    x2vals=[0.6]#[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    t1=int(5/dt)
    t2=int(40/dt)
    deltat=int(10/dt)
    deltax=0.05
    #random.seed(123)

    #CREATE SIMULATION FOLDER
    # if not os.path.exists(SimulationName):
    #     os.makedirs(SimulationName)

    for x2 in x2vals:
        print("Starting dynamics")

        grid=RegularPfc(N,m) # defines environment. grid is a m x N array, m number of maps, N number of units in the network
        
        np.save(SimulationName+"/pfc",grid)
        J=BuildJ(N,grid,J0=J0,nk=N*c,sigma=sigma) # builds connectivity 
        V=np.zeros(N)
        #V=np.random.uniform(0,1,N) # initialize network in a random state, input 
        #V=correlate_activity(grid[0],bump_center=0.5)
        #np.save(SimulationName+"/Vinitial",V)

        s=make_stimulus(maxsteps,N,t1=t1,t2=t2,x1=x1,x2=x2,deltat=deltat,deltax=deltax)

        #np.save(SimulationName+"/Stimuli",s)

        Vvec=dynamics(V,J,s,tautheta=tautheta,tauh=tauh,maxsteps=maxsteps,dt=dt,beta=beta,h0=h0,skipsteps=skipsteps)

        #np.save(SimulationName+"/Vdynamics",Vvec)

        print("Dynamics terminated, result saved")

        plot_heatmap(Vvec,s,maxsteps,x1,x2,t1,t2)
    return

# FUNCTIONS

def plot_heatmap(Vs,S,maxsteps,x1,x2,t1,t2):
    print(x1,x2)
    fig, axs = plt.subplots(2, figsize=(6,3)) 
    #Vs=np.load("1D/Vdynamics.npy")
    #S=np.load("1D/Stimuli.npy")
    print(np.shape(S))
    im = axs[0].imshow(np.log(Vs.T), interpolation='bilinear', cmap=cm.RdYlGn, origin='lower')#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
    im1 = axs[1].imshow(S.T, interpolation='bilinear', cmap=cm.Greys, origin='lower')#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
    
    #ax.axhline(y=400, color='k', linestyle='-')
    #ax.axhline(y=1200, color='k', linestyle='-')
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("top", size="50%", pad=0.3)
    plt.colorbar(im,cax=cax,orientation='horizontal')
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("top", size="50%", pad=0.3)    
    plt.colorbar(im1,cax=cax,orientation='horizontal')
    # ax.set_xticks(np.arange(0,1200,200))
    # ax.set_xticklabels(np.arange(0,1200,200))
    # ax.set_yticks(np.arange(0,1200,200))

    axs[0].set_ylabel("log(V(x))", fontsize=14)
    axs[1].set_xlabel("time", fontsize=14)
    axs[1].set_ylabel("S(x)", fontsize=14)    
    # ax.set_yticklabels(['%.2f'%i for i in np.linspace(0,1,6)]);   
    fig.tight_layout() 
    fig.savefig("heatmap_x1_%.1f_x2_%.1f_t1_%d_t2_%d.png"%(x1,x2,t1,t2))

def make_stimulus(maxsteps,N,t1,t2,x1,x2,deltat,deltax):
    s=np.zeros((maxsteps,N))
    s[t1:t1+deltat,int((x1-deltax)*N):int((x1+deltax)*N)]=1
    s[t2:t2+deltat,int((x2-deltax)*N):int((x2+deltax)*N)]=1
    return s


def K(x1,x2,N,nk=20):
        d0=nk/N
        d=x1-x2
        if d>0.5:
            d=d-1.
        elif d<-0.5:
            d=d+1.
        return np.exp(-abs(d/d0))

def RegularPfc(N,m):
        grid=np.zeros((m,N))
        tempgrid=np.zeros(N)
        for i in range(N):
            tempgrid[i]=float(i)/float(N)
        grid[0][:]=tempgrid    
        for j in range(1,m):
            random.shuffle(tempgrid)
            grid[j][:]=tempgrid
        return grid

def BuildJ(N,grid,nk=20,J0=0.2,sigma=20):
    J=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            for k in range(len(grid)):
                x1=grid[k][i]
                x2=grid[k][j]
                if i!=j:
                    J[i][j]=J[i][j]+K(x1,x2,N,nk=nk)
    return (J-J0)/sigma

def Sigmoid(h,beta,h0):
    return 1./(1.+np.exp(-beta*(h-h0)))        

def dynamics(V,J,s,tautheta=1.,tauh=0.1,maxsteps=1000,dt=0.01,beta=1.,h0=0.,skipsteps=10):
     
        N = len(V)
        Vvec=np.zeros((maxsteps,N))
        theta=np.zeros(N)
        h=np.zeros(N)
        for step in range(maxsteps):
            theta+=dt*(V-theta)/tautheta
            h+=dt*(np.dot(J,V)-h-theta+s[step])/tauh # + random.uniform(0, 10)
            if(step%skipsteps==0):
                 print("h=",np.min(h),np.max(h))
                 print("V=",np.min(V),np.max(V))
            V=Sigmoid(h,beta,h0)
            Vvec[step][:]=V
            #print("Dynamic step: "+str(step)+" done, mean: "+str(np.mean(V))+" sparsity: "+str(pow(np.mean(V),2)/np.mean(pow(V,2))))
        return Vvec

def correlate_activity(pos,bump_center=0.5):
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