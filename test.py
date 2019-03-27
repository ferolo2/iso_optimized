import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg
from matplotlib import pyplot as plt
from ast import literal_eval

from scipy.linalg import eig

import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
sys.path.insert(0,cwd+'/K3df')
#sys.path.insert(0,cwd)

from K3df import K3A, K3B, K3quad
from F3 import F2_alt, Gmatrix, sums_alt as sums, K2i_mat
from F3 import H_mat, F3_mat
import defns, projections as proj, analysis_defns as AD, group_theory_defns as GT


E=5; L=6;
K0 = 40; K1 = 0; K2 = 0; A = 30; B = 20;
#K3 = np.around(K3quad.K3mat(E,L,K0,K1,K2,A,B,'r'),4)

print(defns.shell_list(3.85,L))
print(defns.Emin([1,1,1],L,alpH=-1,xmin=0))
print(defns.shell_breaks(E,L))

#E0_list = defns.E_free_list(L,4,3)
#for e in E0_list:
#  print(e,E0_list[e])
