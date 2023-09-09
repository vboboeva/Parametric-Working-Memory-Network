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
# from helper_make_data import make_data
# from helper_mymax import mymax
# from helper_history import history
# from helper_func import func

def mymax(x1,x2):
	A=np.empty(len(x1), float)
	for i in range(len(x1)):
		if x1[i] > x2[i]:
			A[i]=1
		elif x1[i] < x2[i]:
			A[i]=0
		else:
			x=random.uniform(0,1)
			if x < 0.5:
				A[i]=0
			else:
				A[i]=1		
	return A
