import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg
#from matplotlib import pyplot as plt
#from ast import literal_eval

from scipy.linalg import block_diag

import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
sys.path.insert(0,cwd+'/K3df')

from K3df import K3A, K3B, K3quad
from F3 import F2_alt, Gmatrix, sums_alt as sums, K2i_mat
from F3 import H_mat, F3_mat
import defns, projections as proj, analysis_defns as AD, group_theory_defns as GT

alpha = 0.5
a0 = 0.1; r0=0; P0=0; a2=1e-3
K0=0; K1=0; K2=0; A=0; B=1e2
L = 5
E = 4.21193817  #E1 = 2*omega_1 + 1 = 4.2119381713...
I = 'A1+'

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

# Analytic checks

def const(E,L):
  w1 = sqrt((2*pi/L)**2+1)
  E1 = 2*w1+1
  return 1/(8*L**3*w1**2) * 1/(E-E1)

def F_div(nnp,l1,m1,nnk,l2,m2):
  nnp = list(nnp); nnk = list(nnk)
  if nnp==nnk==[0,0,0]:
    if l1==m1==l2==m2==0:
      return 3
    elif l1==l2==2 and (m1==m2==0 or m1==m2==2):
      return 7.5
    else:
      return 0
  elif nnp==nnk and LA.norm(nnp)==1:
    return defns.ylm(nnp,l1,m1)*defns.ylm(nnp,l2,m2)
  else:
    return 0


def G_div(nnp,l1,m1,nnk,l2,m2):
  nnp = list(nnp); nnk = list(nnk)
  if LA.norm(nnp)==0 and LA.norm(nnk)==1:
    return defns.ylm(nnk,l1,m1)*defns.ylm(nnk,l2,m2)
  elif LA.norm(nnp)==1 and LA.norm(nnk)==0:
    return defns.ylm(nnp,l1,m1)*defns.ylm(nnp,l2,m2)
  elif nnp==[-x for x in nnk] and LA.norm(nnp)==1:
    return defns.ylm(nnp,l1,m1)*defns.ylm(nnp,l2,m2)
  else:
    return 0

def Fmat_div():
  out = np.zeros((42,42))
  nnk_list = [(0,0,0)]+defns.shell_nnk_list([0,0,1])
  for n in range(len(nnk_list)):
    nnk = nnk_list[n]
    for i1 in range(6):
      i = 6*n+i1
      [l1,m1] = defns.lm_idx(i1)
      for i2 in range(6):
        j = 6*n+i2
        [l2,m2] = defns.lm_idx(i2)
        out[i,j] = F_div(nnk,l1,m1,nnk,l2,m2)
  return out

def Gmat_div():
  out = np.zeros((42,42))
  nnk_list = [(0,0,0)]+defns.shell_nnk_list([0,0,1])
  for i in range(42):
    nnp = nnk_list[i//6]
    [l1,m1] = defns.lm_idx(i)
    for j in range(42):
      nnk = nnk_list[j//6]
      [l2,m2] = defns.lm_idx(j)
      out[i,j] = G_div(nnp,l1,m1,nnk,l2,m2)
  return out


F = Fmat_div()
G = Gmat_div()
S = F+G
#print(defns.chop(sorted(LA.eigvals(F).real,key=abs,reverse=True)))
#print(defns.chop(sorted(LA.eigvals(G).real,key=abs,reverse=True)))
#print(defns.chop(sorted(LA.eigvals(S).real,key=abs,reverse=True)))

P = proj.P_irrep_subspace(3.2,5,I) # E=3.2,L=5 is just to specify 2-shell matrix size

F_I = defns.chop(P.T@F@P)
G_I = defns.chop(P.T@G@P)
S_I = defns.chop(P.T@S@P)
#print(S_I)
#print(np.around(F_I,6))
#print(defns.chop(sorted(LA.eigvals(F_I).real,key=abs,reverse=True)))
#print(defns.chop(sorted(LA.eigvals(G_I).real,key=abs,reverse=True)))
#print(defns.chop(sorted(LA.eigvals(S_I).real,key=abs,reverse=True)))



#####################################################
# Numerical checks

if a2==0:
  F = F2_alt.Fmat00(E,L,alpha)
  G = Gmatrix.Gmat00(E,L)
  K2i = K2i_mat.K2inv_mat00(E,L,a0,r0,P0)
  K3 = proj.l0_proj(K3quad.K3mat(E,L,K0,K1,K2,A,B,'r'))
  P = proj.P_irrep_subspace_00(E,L,I)
elif a0==0:
  F = F2_alt.Fmat22(E,L,alpha)
  G = Gmatrix.Gmat22(E,L)
  K2i = K2i_mat.K2inv_mat22(E,L,a2)
  K3 = proj.l2_proj(K3quad.K3mat(E,L,K0,K1,K2,A,B,'r'))
  P = proj.P_irrep_subspace_22(E,L,I)
else:
  F = F2_alt.Fmat(E,L,alpha)
  G = Gmatrix.Gmat(E,L)
  K2i = K2i_mat.K2inv_mat(E,L,a0,r0,P0,a2)
  K3 = K3quad.K3mat(E,L,K0,K1,K2,A,B,'r')
  P = proj.P_irrep_subspace(E,L,I)

S = defns.chop(F+G)
H = defns.chop(F+G+K2i); Hi = defns.chop(LA.inv(H))
FHi = F@Hi;  HiF = Hi@F
FHiF = F@Hi@F
T1 = np.identity(len(F)) - 3*F @ Hi
T2 = np.identity(len(F)) - 3*Hi @ F
F3 = T1 @ F / (3*L**3); F3i = defns.chop(LA.inv(F3))
#F3 = (F/3 - F@Hi@F) / L**3; F3i = defns.chop(LA.inv(F3))

F_I = P.T@F@P
G_I = P.T@G@P
K2i_I = P.T@K2i@P
S_I = F_I+G_I
H_I = F_I+G_I+K2i_I; Hi_I = defns.chop(LA.inv(H_I))
FHi_I = F_I@Hi_I
HiF_I = Hi_I@F_I
FHiF_I = F_I@Hi_I@F_I
T1_I = P.T@T1@P
T2_I = P.T@T2@P
F3_I = P.T@F3@P; F3i_I = defns.chop(LA.inv(F3_I))
K3_I = P.T@K3@P

p = const(E,L)

#nnp=[0,0,0]
#nnk=[0,1,0]
##print(sums.F2KSS(E,L,nnk,0,0,0,0,alpha))
##print(sums.F2KSS(E,L,nnk,2,0,2,0,alpha))
##print(F[0:6,0:6])
#
#F00 = sums.F2KSS(E,L,nnk,0,0,0,0,alpha)
##print(F00,F[6,6]/F00)
#print(F00,F[12,12]/F00)
#F2020 = sums.F2KSS(E,L,nnk,2,0,2,0,alpha)
##print(F2020,F2020/F[9,9],F2020/F00)
#print(F2020,F2020/F[15,15],F2020/F[12,12])
##print(F[6:12,6:12]/F[6,6])
#
#
#for i in range(1,7):
#  print(np.around(F[6*i:6*(i+1),6*i:6*(i+1)]/F[6,6],4))
#  print(np.around(G[0:6,6*i:6*(i+1)]/G[0,6],4))
#
##print(Gmatrix.G(E,L,[0,0,0],[0,0,1],0,0,0,0))
##print(Gmatrix.G(E, L, nnp, nnk,2,0,2,0))#/Gmatrix.G(E,L,[0,0,0],[0,0,1],0,0,0,0))



#print(sorted(LA.eigvals(F_I).real,key=abs,reverse=True))
#print(sorted(LA.eigvals(G_I).real,key=abs,reverse=True))
#print(sorted(LA.eigvals(K2i_I).real,key=abs,reverse=True))
#print(sorted(LA.eigvals(S_I).real,key=abs,reverse=True))
#print(sorted(LA.eigvals(H_I).real,key=abs,reverse=True))
#print(sorted(LA.eigvals(Hi_I).real,key=abs))
#print(sorted(LA.eigvals(FHi_I).real,key=abs,reverse=True))
#print(sorted(LA.eigvals(FHiF_I).real,key=abs,reverse=True))
#print(sorted(LA.eigvals(F3_I).real,key=abs,reverse=True))
print(sorted(LA.eigvals(F3i_I).real,key=abs))
print(sorted(LA.eigvals(K3_I).real,key=abs,reverse=True))
print(sorted(LA.eigvals(F3i_I+K3_I).real,key=abs))
#print(sorted(LA.eigvals(np.identity(len(F3_I))+F3_I@K3_I).real,key=abs))

#print(sorted(LA.eigvals(F).real,key=abs,reverse=True))
#print(sorted(LA.eigvals(G).real,key=abs,reverse=True))
#print(sorted(LA.eigvals(S).real,key=abs,reverse=True))
print(sorted(LA.eigvals(F3).real,key=abs,reverse=True))
print(sorted(LA.eigvals(K3).real,key=abs,reverse=True)[0:11])
print(sorted(LA.eigvals(F3i+K3).real,key=abs))


[F3_elist,F3_vlist] = LA.eig(F3i_I)
F3_ivec = [i for i,e in enumerate(F3_elist) if abs(e)<1e-2]


[K3_elist,K3_vlist] = LA.eig(K3_I)
K3_ivec = [i for i,e in enumerate(K3_elist) if abs(e)>1e-2]
print()
for i in F3_ivec:
  e_F3 = F3_elist[i].real
  v_F3 = F3_vlist[:,i]
  for j in K3_ivec:
    e_K3 = K3_elist[j].real
    v_K3 = K3_vlist[:,j]
    #print(v_F3)
    print(e_F3,e_K3,abs(np.dot(v_F3,v_K3)))


#print(F_I/p)
#print(F3_I)
#print(K3_I)
#print((F3i_I+K3_I)/p)

#print(p)


#####################################################################


def x2(E,L,nnk):
  k = LA.norm(nnk)*2*pi/L
  return defns.qst(E,k)**2*(L/(2*pi))**2

def r2(E,L,nnk,nna):
  nnk = np.array(nnk); nna = np.array(nna)
  nk = LA.norm(nnk); k = nk*2*pi/L
  if nk==0:
    nnk_hat = np.array([0,0,0])
  else:
    nnk_hat = nnk/LA.norm(nnk)
  g = defns.gamma(E,k)
  r = nna + nnk_hat * (np.dot(nna,nnk_hat)*(1/g-1) + nk/(2*g))
  return LA.norm(r)**2

def b2(E,L,nnp,nnk):
  nnp = np.array(nnp); nnk = np.array(nnk)
  p = 2*pi/L*LA.norm(nnp); k = 2*pi/L*LA.norm(nnk)
  omp = defns.omega(p); omk = defns.omega(k)
  return (E-omp-omk)**2 - (2*pi/L*LA.norm(nnp+nnk))**2

#nnk = [0,0,1]
#
#xx = x2(E,L,nnk)
#print(xx)
#for nna in [[0,0,0],[0,0,1]]:
#  rr = r2(E,L,nnk,nna)
#  print(nnk,nna,rr,xx-rr)
#  bb = b2(E,L,nna,nnk)
#  print(nnk,nna,bb,bb-1)
#
##print(sums.hh(E,0),sums.hh(E,2*pi/L))
#print(Gmatrix.boost(np.array([0,0,0]),np.array([0,0,2*pi/L]),E))
#print(Gmatrix.boost(np.array([0,0,2*pi/L]),np.array([0,0,0]),E))
#print(Gmatrix.boost(np.array([0,0,2*pi/L]),np.array([0,0,-2*pi/L]),E))
