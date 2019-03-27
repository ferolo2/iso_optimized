from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
from F3 import H_mat
from F3 import sums_alt as sums
from F3 import F2_alt as F2
from F3 import K2i_mat as K2
from F3 import Gmatrix as Gm
import defns
import math
import numpy as np
import time
from numba import jit,njit,prange,autojit,vectorize
from F3 import sums_alt as sums
from scipy.special import erfc,erfi,erf
from scipy.optimize import fsolve
from joblib import Parallel, delayed
from numpy.lib.scimath import sqrt





E =2.726795460408

listE = np.linspace(2*np.sqrt(1.-1/36.)+0.98,2*np.sqrt(1.-1/36.)+1,31)
listE = np.linspace(2.56,2.62,60)
print(listE)
listL = np.array(range(5,6,1))*1.0
nnk = np.array([0.,0.,0.])
a0 = 2.


for L in listL:
    k = sums.norm(nnk*2*math.pi/L)
    sols = []
    for E in listE:
        omk = np.sqrt(1+k**2)
        E2s = np.sqrt((E**2+1)-2*E*omk)
        q2s = defns.qst2(E,k)
        hhk = sums.hh(E, k)
        f2 =sums.F2KSS(E,L,np.array(nnk),0,0,0,0,0.5)
        sols.append([E,32*math.pi*E2s*omk*f2 + np.sqrt(abs(q2s))*(1-hhk)-1/a0])

    sols = np.array(sols)
    print(sols[:,1])

    f3b = interp1d(sols[:,0], np.real(sols[:,1]), kind='cubic')
    Esol = fsolve(f3b,listE[30])[0]
    print(L,Esol)
    print(L,np.sqrt((Esol - omk)**2-(k)**2),L)
#print(f2)
