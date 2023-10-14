import os
import numpy as np
import sys

''' SCRIPT USED TO ANALYSE NETWORK DATA  '''
''' PRODUCES FIGURES IN THE MANUSCRIPT BOBOEVA ET AL 2023 PUBLISHED IN ELIFE'''


# ''' PRODUCES FIG 2A '''

# tauhH=0.5
# tauthetaH=7.5
# DH=0.3
# AHtoWM=0.5
# eps=0
# N=2000
# num_sims=100
# ITI=5
# psycholabel=True

# os.system('python psychometric_fTISI.py %s %s %s %s %s %s %s %s %s'%(tauhH, tauthetaH, DH, AHtoWM, eps, N, num_sims, ITI, psycholabel))


# ''' FIG 2C and 7B (NETWORK ONLY)'''

# tauhH=0.5
# tauthetaH=7.5
# DH=0.3
# AHtoWM=0.5
# eps=0
# N=2000
# num_sims=100
# ISI='all' # choose btw 2,6,10 and all
# ITI=5 # choose btw 1.2, 5 and 10 (this is not the real ITI, only the part of ITI after each trial, there is also the part of ITI before the beginning of each trial that is 1 sec. So to get real ITI we add 1 sec to this value)
# psycholabel=False

# os.system('python frac_class.py %s %s %s %s %s %s %s %s %s %s'%(tauhH, tauthetaH, DH, AHtoWM, eps, N, num_sims, ISI, ITI, psycholabel))


# ''' FIG 2D (NETWORK ONLY) and 7E'''

# tauhH=0.5
# tauthetaH=7.5
# DH=0.3
# AHtoWM=0.5
# eps=0
# N=2000
# num_sims=100
# ISI='all' # choose btw 2,6,10 and all
# ITI=5 # choose btw 1.2, 5 and 10
# psycholabel=False

# os.system('python history_matrix.py %s %s %s %s %s %s %s %s %s %s'%(tauhH, tauthetaH, DH, AHtoWM, eps, N, num_sims, ISI, ITI, psycholabel))

# ''' FIG 2E (NETWORK ONLY)'''

# tauhH=0.5
# tauthetaH=7.5
# DH=0.3
# eps=0
# N=2000
# num_sims=100
# ISI='all' # choose btw 2,6,10 and all
# ITI=5
# psycholabel=True

# os.system('python psychometric_opto.py %s %s %s %s %s %s %s %s %s'%(tauhH, tauthetaH, DH, eps, N, num_sims, ISI, ITI, psycholabel))


# ''' FIG 2F (NETWORK ONLY)'''

# tauhH=0.5
# tauthetaH=7.5
# DH=0.3
# eps=0
# N=2000
# num_sims=100
# ISI='all' # choose btw 2,6,10 and all
# ITI=5
# psycholabel=False

# os.system('python bias_opto.py %s %s %s %s %s %s %s %s %s'%(tauhH, tauthetaH, DH, eps, N, num_sims, ISI, ITI, psycholabel))


# ''' FIG 3B '''

# tauhH=0.5
# tauthetaH=7.5
# DH=0.3
# AHtoWM=0.5
# eps=0
# N=2000
# num_sims=1
# ISI='all' # choose btw 2,6,10 and all
# ITI=5

# os.system('python plot_locbump_trials.py %s %s %s %s %s %s %s %s %s'%(tauhH, tauthetaH, DH, AHtoWM, eps, N, num_sims, ISI, ITI))


# ''' FIG 3C '''

# tauhH=0.5
# tauthetaH=7.5
# DH=0.3
# AHtoWM=0.5
# eps=0
# N=2000
# num_sims=1
# ISI='all' # choose btw 2,6,10 and all
# ITI=5

# os.system('python barplot_bumploc_data_curr_1back_2back.py %s %s %s %s %s %s %s %s %s'%(tauhH, tauthetaH, DH, AHtoWM, eps, N, num_sims, ISI, ITI))


# ''' FIG 3E '''

# tauhH=0.5
# tauthetaH=7.5
# DH=0.3
# AHtoWM=0.5
# eps=0
# N=2000
# num_sims=100
# ITI=5
# psycholabel=False

# os.system('python cbias_bar_fTISI.py %s %s %s %s %s %s %s %s %s'%(tauhH, tauthetaH, DH, AHtoWM, eps, N, num_sims, ITI, psycholabel))


# ''' FIG 3F for all ISI '''

# tauhH=0.5
# tauthetaH=7.5
# DH=0.3
# AHtoWM=0.5
# eps=0
# N=2000
# num_sims=100
# ISI='all'
# ITI=5
# psycholabel=False

# os.system('python bias_3back.py %s %s %s %s %s %s %s %s %s %s'%(tauhH, tauthetaH, DH, AHtoWM, eps, N, num_sims, ISI, ITI, psycholabel))

# ''' FIG 3F '''

# tauhH=0.5
# tauthetaH=7.5
# DH=0.3
# AHtoWM=0.5
# eps=0
# N=2000
# num_sims=100
# ITI=5
# psycholabel=False

# os.system('python bias_3back_fTISI.py %s %s %s %s %s %s %s %s %s'%(tauhH, tauthetaH, DH, AHtoWM, eps, N, num_sims, ITI, psycholabel))


# ''' FIG 3G '''

# tauhH=0.5
# tauthetaH=7.5
# DH=0.3
# AHtoWM=0.5
# eps=0
# N=2000
# num_sims=100
# ISI='all'
# ITI=5
# psycholabel=False

# os.system('python bias_jump.py %s %s %s %s %s %s %s %s %s %s'%(tauhH, tauthetaH, DH, AHtoWM, eps, N, num_sims, ISI, ITI, psycholabel))


# ''' FIG 4B '''

# tauhH=0.5
# tauthetaH=7.5
# DH=0.3
# AHtoWM=0.5
# eps=0
# N=2000
# num_sims=100
# ITI=5

# os.system('python plot_distrib_bumps.py %s %s %s %s %s %s %s %s'%(tauhH, tauthetaH, DH, AHtoWM, eps, N, num_sims, ITI))

# ''' FIG 7A '''

# tauhH=0.5
# tauthetaH=7.5
# DH=0.3
# AHtoWM=0.5
# eps=0
# N=2000
# num_sims=100
# ISI='all'
# psycholabel=True

# os.system('python psychometric_fTITI.py %s %s %s %s %s %s %s %s %s'%(tauhH, tauthetaH, DH, AHtoWM, eps, N, num_sims, ISI, psycholabel))

# ''' FIG 7C '''

# tauhH=0.5
# tauthetaH=7.5
# DH=0.3
# AHtoWM=0.5
# eps=0
# N=2000
# num_sims=100
# ISI='all'
# psycholabel=False

# os.system('python cbias_bar_fTITI.py %s %s %s %s %s %s %s %s %s'%(tauhH, tauthetaH, DH, AHtoWM, eps, N, num_sims, ISI, psycholabel))


# ''' FIG 7D '''

# tauhH=0.5
# tauthetaH=7.5
# DH=0.3
# AHtoWM=0.5
# eps=0
# N=2000
# num_sims=100
# ISI='all'
# psycholabel=False

# os.system('python bias_3back_fTITI.py %s %s %s %s %s %s %s %s %s'%(tauhH, tauthetaH, DH, AHtoWM, eps, N, num_sims, ISI, psycholabel))


# ''' REDUCTION OF CONTRACTION BIAS WITH OPTO MANIPULATION - THIS IS NOT IN THE PAPER '''

# tauhH=0.5
# tauthetaH=7.5
# DH=0.3
# eps=0
# N=2000
# num_sims=100
# ISI='all'
# ITI=5
# psycholabel=False

# os.system('python cbias_bar_opto.py %s %s %s %s %s %s %s %s %s'%(tauhH, tauthetaH, DH, eps, N, num_sims, ISI, ITI, psycholabel))

''' FIG 7F '''

tauhH=0.5
tauthetaH=7.5
DH=0.3
AHtoWM=0.5
eps=0
N=2000
num_sims=100
ISI='all'
psycholabel=False

os.system('python compute_bias_3back_bars.py %s %s %s %s %s %s %s %s %s'%(tauhH, tauthetaH, DH, AHtoWM, eps, N, num_sims, ISI, psycholabel))

os.system('python plot_bias_3back_bars.py %s %s %s %s %s %s %s %s %s'%(tauhH, tauthetaH, DH, AHtoWM, eps, N, num_sims, ISI, psycholabel))