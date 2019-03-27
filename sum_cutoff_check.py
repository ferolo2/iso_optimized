import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg
#from matplotlib import pyplot as plt
#from ast import literal_eval

from scipy.linalg import eig

import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
sys.path.insert(0,cwd+'/K3df')

from K3df import K3A, K3B, K3quad
from F3 import F2_alt, Gmatrix, sums_alt as sums, K2i_mat
from F3 import H_mat, F3_mat
import defns, projections as proj, analysis_defns as AD, group_theory_defns as GT


E = 3.84; L=5; alpha=0.5
nnk=np.array([1,1,0])
k = 2*pi/L*LA.norm(nnk)
print(sums.hh(E,k))

S0 = np.zeros((6,6))
S1 = S0
for i in range(6):
  for j in range(6):
    [l1,m1] = defns.lm_idx(i)
    [l2,m2] = defns.lm_idx(j)
    
    t0 = time.time()
    S0[i,j] = sums.sum_nnk(E, L, nnk,l1,m1,l2,m2,alpha,smart_cutoff=0)
    t1 = time.time(); print('old time:',t1-t0)
    S1[i,j] = sums.sum_nnk(E, L, nnk,l1,m1,l2,m2,alpha,smart_cutoff=1)
    print('new time:',time.time()-t1)

print(S0)
print(S0-S1)
