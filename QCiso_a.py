from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
    import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
sys.path.insert(0,cwd+'/K3df')
from F3 import H_mat,F3_mat,Gmatrix
import itertools
import math
import numpy as np
import time
import projections as pr
from numba import jit,njit,prange,autojit
from multiprocessing import Pool, Process
import multiprocessing
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from K3df import K3cubicB as K33B
from K3df import K3quad as K3
from defns import list_nnk,chop
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib
#from matplotlib import colors


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def F3i_00(e,L,a0,IPV):

  f3mat= F3_mat.F3mat00(e,L,a0,0.,0.,0.,0.5,IPV)
#  resaux = sorted(np.linalg.eig(pr.iso_proj00(f3mat,e,L))[0],key=abs)
  return 1./(np.ones(len(f3mat))@f3mat@np.ones(len(f3mat)))



matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"].append(r'\usepackage{xfrac}')

fig, ax = plt.subplots()

ax.set_ylabel('$\\frac{1}{F^{iso}}$  ', fontsize=24,rotation=0,labelpad=29)
ax.set_xlabel('$m a_0$', fontsize=21)
plt.gcf().subplots_adjust(left=0.2)
plt.gcf().subplots_adjust(bottom=0.15)

#plt.xlim([-2,0])
#plt.ylim([-20000,20000])

#print("Number of cpu : ", multiprocessing.cpu_count())

rangeL= [20,30,33 ]
IPV=0
Etest =2.9991
a0=1
da = 0.01
nmax = 100 #200/4

Arange = np.linspace(a0, a0 + da*(nmax-1),num=nmax,endpoint=True)
Arange = -1*np.logspace(np.log10(a0),np.log10(10.),nmax)

#Arange = [2/np.sqrt(3+2*Etest-Etest**2)]
lenA = len(list(Arange))



#print(Arange)
#exit()

aux = []
for L in rangeL:                                                                                                                                                    
        aux.append((Etest,L,Arange,0.,0.,0.,0.5,IPV))   

start = time.time() 
pool = Pool(processes=4)
allresult = pool.starmap(F3_mat.F3mat00iso, aux)
pool.close
end = time.time()
print('time is:', end - start, ' s')

oneF = []
for i in range(len(rangeL)):
#    print(allresult[i])
#    oneF.append(allresult[i][0])
    
    ax.plot(Arange,allresult[i],'.',label='$E=%5.3f$, $I_{PV}=-1, L=%d$' % (Etest,rangeL[i]), linewidth=2)
#ax.plot(Arange,allresult[1],'.',label='$E=%5.3f$, $I_{PV}=-1, L=%d$' % (Etest,L), linewidth=2)

#for L in rangeL:
#    start = time.time()
#    result = F3_mat.F3mat00iso(Etest,L,Arange,0.,0.,0.,0.5,IPV)
#    end = time.time()
#    print('time is:', end - start, ' s')
#    ax.plot(Arange,result,'.',label='$E=%5.3f$, $I_{PV}=-1, L=%d$' % (Etest,L), linewidth=2)

#pool.close
#ax.plot(rangeL, np.array(oneF) ,'o',color='black')
ax.plot(Arange,0*Arange,'--',color='black')
ax.plot(Arange,60000+ 0*Arange,'--',color='black')
plt.axvline(x=2/np.sqrt(3+2*Etest-Etest**2),linestyle='--',color='magenta',label='particle+dimer')

    
ax.legend(loc='lower left', fontsize=10, ncol=1, numpoints=1)



plt.tight_layout()
plt.savefig('Fiso_a0_E299_2.pdf',bbox_inches='tight', pad_inches = 0) 
plt.show()
  
exit()

