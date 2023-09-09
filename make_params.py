#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 8 Feb

@author: vb
"""
import numpy as np
#import random
import os
#import time
#import cProfile
#import pstats
#import sys
#from numba import jit


numsims=10

tauhHvals=[0.5]
tauthetaHvals=[7.5]
DHvals=[0.25,0.27,0.3]
AHtoWMvals=[0.5]
ITIvals=[1.2,5,10]
params1=np.zeros((len(tauhHvals)*len(tauthetaHvals)*len(DHvals)*len(AHtoWMvals)*len(ITIvals)*numsims,6))
o=0

for i in range(len(tauhHvals)):
    for j in range(len(tauthetaHvals)):
        for k in range(len(DHvals)):
            for l in range(len(AHtoWMvals)):
                for m in range(len(ITIvals)):
                    for n in range(numsims):
                        FolderName="AHtoWM%.2f_tauhH%.2f_tauthetaH%.2f_DH%.2f_eps0.00_ITI%.1f"%(AHtoWMvals[l],tauhHvals[i],tauthetaHvals[j],DHvals[k],ITIvals[m])
                        #print(FolderName)
                        try:
                            A=np.load( FolderName+"/inf_sim%d.npy"%n)
                            B=np.load( FolderName+"/VWM_sim%d.npy"%n)
                            C=np.load( FolderName+"/VH_sim%d.npy"%n)
                            print("Exists")
                        except:
                            print("Don't exist")
                            params1[o,0]=tauhHvals[i]
                            params1[o,1]=tauthetaHvals[j]
                            params1[o,2]=DHvals[k]
                            params1[o,3]=AHtoWMvals[l]
                            params1[o,4]=ITIvals[m]
                            params1[o,5]=n
                            o+=1
			
params1=params1[:o]
np.savetxt("params.txt", params1, fmt='%.2f')

