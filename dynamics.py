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

    SimulationName="1D"
    N=1000 #number of neurons
    m=1 #number of maps

    tauVWM=0.1
    tauVH=1.
    tautheta=5.
    tauF=10.
    U=0.5

    maxsteps=3000
    skipsteps=1
    dt=0.01

    beta=10.
    h0=0.
    J0=0.2
    c=0.03 # sparsity of the non-zero connections 
    sigma=30

    num_sims=1
    random.seed(time.time)

    x1vals=np.array([random.uniform(0.1, 0.9) for i in range(num_sims)])
    x2vals=np.array([random.uniform(0.1, 0.9) for i in range(num_sims)])

    t1=int(0/dt)
    t2=int(5/dt)
    deltat=int(2/dt)
    deltax=0.05

    #CREATE SIMULATION FOLDER
    if not os.path.exists(SimulationName):
        os.makedirs(SimulationName)

    labels=np.zeros(len(x1vals))
    for i in range(len(x1vals)):
        if (x1vals[i]-x2vals[i] >= 0. ):
            labels[i]=1
        else:
            labels[i]=0

    np.save(SimulationName+"/labels",labels)

    RingWM=MakeRing(N,m) # defines environment. Ring is a m x N array, m number of maps, N number of units in the network
    JWMtoWM=BuildJ(N,RingWM,J0=J0,nk=N*c,sigma=sigma) # builds connectivity within WM net

    #np.save(SimulationName+"/pfc",Ring1)

    RingH=MakeRing(N,m)
    JHtoH=BuildJ(N,RingH,J0=J0,nk=N*c,sigma=sigma) # builds connectivity within H net

    # no need to make within network connectivity, as they are one-to-one    

    for i in range(len(x1vals)):
        x1=x1vals[i]
        x2=x2vals[i]
        print(i,x1,x2)
        
        VWM=np.zeros(N)
        VH=np.zeros(N)
        # initialize network in a random state
        #V=np.random.uniform(0,1,N)  
        # initialize network in the center
        #V=CorrAct(Ring[0],bump_center=0.5) 
        #np.save(SimulationName+"/Vinitial",V)

        s=MakeStim(maxsteps,N,t1=t1,t2=t2,x1=x1,x2=x2,deltat=deltat,deltax=deltax)

        #np.save(SimulationName+"/Stimuli",s)

        VWMsave, VHsave = UpdateNet(VWM,JWMtoWM,VH,JHtoH,s,tautheta=tautheta,tauVWM=tauVWM,tauVH=tauVH,tauF=tauF,U=U,maxsteps=maxsteps,dt=dt,beta=beta,h0=h0,skipsteps=skipsteps)

        if (i==0):
            data=np.ravel(VWMsave, 'F')
        else:
            data=np.vstack((data,np.ravel(VWMsave, 'F')))

        PlotHeat(VWMsave,VHsave,s,maxsteps,x1,x2,t1,t2)

    print(np.shape(data))

    np.save(SimulationName+"/data", data)

    print("Dynamics terminated, result saved")

    return

# FUNCTIONS

def PlotHeat(VWMs,VHs,S,maxsteps,x1,x2,t1,t2):
    print(x1,x2)
    fig, axs = plt.subplots(3, figsize=(9,6)) 
    #Vs=np.load("1D/Vdynamics.npy")
    #S=np.load("1D/Stimuli.npy")
    print(np.shape(S))
    im = axs[0].imshow(VWMs.T, interpolation='bilinear', cmap=cm.RdYlGn, origin='lower')#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
    im1 = axs[1].imshow(VHs.T, interpolation='bilinear', cmap=cm.RdYlGn, origin='lower')#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
    im2 = axs[2].imshow(S.T, interpolation='bilinear', cmap=cm.Greys, origin='lower')#,vmax=abs(Vs).max(), vmin=-abs(Vs).max())
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
    axs[2].set_ylabel("S(x)", fontsize=14)    
    axs[1].set_xlabel("time", fontsize=14)
    # ax.set_yticklabels(['%.2f'%i for i in np.linspace(0,1,6)]);   
    fig.tight_layout() 
    fig.savefig("heatmap_x1_%.2f_x2_%.2f_t1_%d_t2_%d.png"%(x1,x2,t1,t2))

def MakeStim(maxsteps,N,t1,t2,x1,x2,deltat,deltax):
    s=np.zeros((maxsteps,N))
    s[t1:int(t1+deltat),int((x1-deltax)*N):int((x1+deltax)*N)]=1
    s[t2:int(t2+deltat),int((x2-deltax)*N):int((x2+deltax)*N)]=1
    return s


def K(x1,x2,N,nk=20):
        d0=nk/N
        d=x1-x2
        if d>0.5:
            d=d-1.
        elif d<-0.5:
            d=d+1.
        return np.exp(-abs(d/d0))

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

def BuildJ(N,grid,nk=20,J0=0.2,sigma=1):
    J=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            for k in range(len(grid)):
                x1=grid[k][i]
                x2=grid[k][j]
                if i!=j:
                    J[i][j]=J[i][j]+K(x1,x2,N,nk=nk) 
    return (J-J0)/sigma

def Logistic(h,beta,h0):
    return 1./(1.+np.exp(-beta*(h-h0)))        

def UpdateNet(VWM,JWMtoWM,VH,JHtoH,s,tautheta=1.,tauVWM=0.1,tauVH=10,tauF=10.,U=0.1,maxsteps=1000,dt=0.01,beta=1.,h0=0.,skipsteps=10):
     
        N = len(VWM)
        #Vvec=np.zeros((maxsteps,N))
        VWMsave=np.zeros((int(maxsteps/skipsteps+1),N))
        VHsave=np.zeros((int(maxsteps/skipsteps+1),N))

        #thetavec=np.zeros((maxsteps,N))
        #uvec=np.zeros((maxsteps,N))

        theta=np.zeros(N)
        hWM=np.zeros(N)
        hH=np.zeros(N)
        u=np.zeros(N)

        k=0
        for step in range(maxsteps):
            theta+=dt*(0.3*VWM-theta)/tautheta
            #u+=dt*(-(u-U)+tauF*U*VWM*(1.-u))            
            
            # UPDATE THE WM NET
            hWM = np.dot(JWMtoWM,VWM) + VH/3. + s[step] - theta #+ #random.uniform(0., 0.1)
            VWM += dt*(Logistic(hWM,beta,h0)-VWM)/tauVWM

            # UPDATE THE H NET
            hH = np.dot(JHtoH,VH) + VWM/3. #+ random.uniform(0., 2.)
            VH += dt*(Logistic(hH,beta,h0)-VH)/tauVH


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