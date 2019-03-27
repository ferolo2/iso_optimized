import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
sys.path.insert(0,cwd+'/K3df')
from F3 import H_mat,F3_mat,Gmatrix
import math
import numpy as np
import time
import projections as pr
from numba import jit,njit,prange,autojit
from multiprocessing import Pool
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from K3df import K3cubicB as K33B
from defns import list_nnk,chop




data=np.array([[ 2.73800000e+000,  5.19822603e-167],
      [ 2.74000000e+000,  2.16721667e-167],
      [ 2.74200000e+000,  9.42286648e-172],
      [ 2.74400000e+000,  2.51244956e-172],
      [ 2.74600000e+000,  4.47844044e-173],
      [ 2.74800000e+000, -8.67390877e-174],
      [ 2.75000000e+000, -1.67823740e-173],
      [ 2.75200000e+000, -1.35773597e-173],
      [ 2.75400000e+000, -9.03189522e-174],
      [ 2.75600000e+000, -5.52053483e-174],
      [ 2.75800000e+000, -3.22464530e-174],
      [ 2.76000000e+000, -1.83312030e-174]])


data=np.array([[ 2.74400000e+000,  3.80757307e-179],
               [ 2.74600000e+000,  6.52750154e-180],
               [ 2.74800000e+000, -4.19487395e-180],
               [ 2.75000000e+000, -6.52055204e-180]])


EN = data[:,0]
EV10 =data[:,1]*10**(174)

f3b = interp1d(EN, EV10, kind='quadratic')

print(fsolve(f3b,2.74600000)[0])



print(data)
