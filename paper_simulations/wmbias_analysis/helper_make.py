import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mat4py import loadmat
import scipy as sp
from matplotlib import cm
from helper_make_data import make_data 
from helper_scatter import compute_scatter
from matplotlib.ticker import FormatStrFormatter
import sys
import os


def make(whichdelay, whichITI, delay, ITIs):

	# define a mask for the delay criterion
	if whichdelay == 'all':
		mask_delay = np.ones(len(delay), dtype=bool)
	else:
		mask_delay = delay == int(whichdelay)


	# define a mask for the ITI criterion
	if whichITI == 'all':
		mask_ITI = np.ones(len(ITIs), dtype=bool)

	elif whichITI == 'low':
		threshold_low = np.percentile(ITIs, 33)
		mask_ITI = ITIs < threshold_low

	elif whichITI == 'mid':
		threshold_low = np.percentile(ITIs, 33)
		threshold_high = np.percentile(ITIs, 66)
		mask_ITI = (ITIs >= threshold_low) & (ITIs < threshold_high)

	elif whichITI == 'high':
		threshold_high = np.percentile(ITIs, 66)
		mask_ITI = (ITIs >= threshold_high)

	# get the indices of the combined masks
	ids = np.where( mask_delay & mask_ITI )[0]
	
	return ids