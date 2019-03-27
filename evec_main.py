import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg
from ast import literal_eval


import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
sys.path.insert(0,cwd+'/K3df')
#sys.path.insert(0,cwd)

from F3 import F2_alt, Gmatrix, sums_alt as sums, K2i_mat
from F3 import H_mat, F3_mat
import defns, projections as proj, analysis_defns as AD, group_theory_defns as GT


alpha=0.5

a0=0.1; a2=0;    r0=0; P0=0

E=4.24216326; L=5

I='E+'


if a2==0:
  F3 = F3_mat.F3mat00(E,L,a0,r0,P0,a2,alpha)
  F3_I = proj.irrep_proj_00(F3,E,L,I)
elif a0==r0==P0==0:
  F3 = F3_mat.F3mat22(E,L,a0,r0,P0,a2,alpha)
  F3_I = proj.irrep_proj_22(F3,E,L,I)
else:
  F3 = F3_mat.F3mat(E,L,a0,r0,P0,a2,alpha)
  F3_I = proj.irrep_proj(F3,E,L,I)

F3i_I = defns.chop(LA.inv(F3_I))
w_list,v_list = LA.eig(F3i_I)

F3i = defns.chop(LA.inv(F3))


w0,v0 = AD.smallest_eig_evec(F3i)  # Need to make evec_decomp() for 00 and 22 cases

proj.evec_decomp(v0,E,L,I)

s = 0
for I in GT.irrep_list():
  s += proj.evec_decomp(v0,E,L,I)
print('Total:',s)

