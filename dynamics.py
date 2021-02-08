#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:31:07 2019

@author: davide
"""
import numpy as np
import random
import os
import matplotlib.pyplot as plt


def main():

    SimulationName="1D"
    N=1000 #number of neurons
    m=1 #number of maps
    gamma=10. #
    L=50.0 #size of the map
    tauh=0.1
    tautheta=1.
    maxsteps=1000
    skipsteps=10
    dt=0.001
    beta=1.
    h0=0.
    J0=0.2


    random.seed(123)

    #CREATE SIMULATION FOLDER
    if not os.path.exists(SimulationName):
        os.makedirs(SimulationName)

    print("Starting dynamics")

    grid=RegularPfc(N,L,m) # defines environment. grid is a m x N array, m number of maps, N number of units in the network
    
    np.save(SimulationName+"/pfc",grid)
    J=BuildJ(N,grid,L) # builds connectivity 
    #V=np.random.uniform(0,1,N) # initialize network in a random state, input 
    V=correlate_activity(grid[0],L,L/2)
    np.save(SimulationName+"/Vinitial",V)

    Vvec=dynamics(V,J,tautheta=tautheta,tauh=tauh,maxsteps=maxsteps,dt=dt,beta=beta,h0=h0,J0=J0,skipsteps=skipsteps)

    np.save(SimulationName+"/Vdynamics",Vvec)

    print("Dynamics terminated, result saved")
    return

# FUNCTIONS
def Mex_Hat(x1,x2,L,gamma):
        d=x1-x2
        if d>float(L)/2.0:
            d=d-L
        elif d<-float(L)/2.0:
            d=d+L
        return (np.exp(-abs(d)) - np.exp(-abs(d)/gamma)/gamma)/(1.-1./gamma) # +gamma*np.sign(d)*np.exp(-abs(d))
def K(x1,x2,L):
        d=x1-x2
        if d>float(L)/2.0:
            d=d-L
        elif d<-float(L)/2.0:
            d=d+L
        return np.exp(-abs(d))
def RegularPfc(N,L,m):
        grid=np.zeros((m,N))
        tempgrid=np.zeros(N)
        for i in range(N):
            tempgrid[i]=i*float(L)/float(N)
        grid[0][:]=tempgrid    
        for j in range(1,m):
            random.shuffle(tempgrid)
            grid[j][:]=tempgrid
        return grid

def BuildJ(N,grid,L):
    J=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            for k in range(len(grid)):
                x1=grid[k][i]
                x2=grid[k][j]
                if i!=j:
                    J[i][j]=J[i][j]+K(x1,x2,L)
    return J

def Sigmoid(h,beta,h0):
    return 1./(1.+np.exp(-beta*(h-h0)))        

def dynamics(V,J,tautheta=1.,tauh=0.1,maxsteps=1000,dt=0.01,beta=1.,h0=0.,J0=0.2,skipsteps=10):
     
        N = len(V)
        Vvec=np.zeros((maxsteps,N))
        theta=np.zeros(N)
        h=np.zeros(N)
        for step in range(maxsteps):
            theta+=dt*(V-theta)/tautheta
            #h=h+dt*(np.dot(J-J0,V)-h)/tauh # + random.uniform(0, 10)
            
            h=np.dot(J-J0,V)
            if(step%skipsteps==0):
                print("h=",np.min(h),np.max(h))
                print("V=",Sigmoid(np.min(h),beta,h0),Sigmoid(np.max(h),beta,h0))
            V=Sigmoid(h,beta,h0)
            Vvec[step][:]=V
            #print("Dynamic step: "+str(step)+" done, mean: "+str(np.mean(V))+" sparsity: "+str(pow(np.mean(V),2)/np.mean(pow(V,2))))
        return Vvec

def correlate_activity(pos,L,center):
    V=np.zeros(len(pos))
    for i in range(len(V)):
        V[i]= K(pos[i],center,L)
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