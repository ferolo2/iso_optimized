import math
import numpy as np
#sqrt=np.sqrt;
pi=np.pi
from itertools import permutations as perms
from scipy.linalg import block_diag

import sums_alt as sums
from defns import list_nnk, shell_list, shell_nnk_list, perms_list, chop, full_matrix
from group_theory_defns import Dmat
from projections import l2_proj

from numba import jit,njit

@jit(nopython=True,fastmath=True,cache=True)
def sqrt(x):
    return np.sqrt(x)

@jit(nopython=True,fastmath=True,cache=True)
def myabs(x):
    return abs(x)


########################################################################
# Construct full F-tilde matrix
# Symmetries are exploited to speed up computation
########################################################################

# TB: make 6x6 F-tilde matrix for a given nnk
def Fmat_k(E,L,nnk,alpha):
  # k=(0,0,0)
  if nnk==[0,0,0]:
    a = sums.F2KSS(E,L,nnk,0,0,0,0,alpha)
    b = sums.F2KSS(E,L,nnk,2,-2,2,-2,alpha)
    c = sums.F2KSS(E,L,nnk,2,0,2,0,alpha)
    return np.diag([a,b,b,c,b,c])

  # k=(0,0,a)
  elif nnk[0]==nnk[1]==0:
    a = sums.F2KSS(E,L,nnk,0,0,0,0,alpha)
    b = sums.F2KSS(E,L,nnk,2,-2,2,-2,alpha)
    c = sums.F2KSS(E,L,nnk,2,-1,2,-1,alpha)
    d = sums.F2KSS(E,L,nnk,2,0,2,0,alpha)
    e = sums.F2KSS(E,L,nnk,2,2,2,2,alpha)
    f = sums.F2KSS(E,L,nnk,0,0,2,0,alpha)

    out = np.diag([a,b,c,d,c,e])
    out[0][3] = f; out[3][0] = f
    return out

  # k=(a,a,0)
  elif nnk[0]==nnk[1]!=0 and nnk[2]==0:
    a = sums.F2KSS(E,L,nnk,0,0,0,0,alpha)
    b = sums.F2KSS(E,L,nnk,2,-2,2,-2,alpha)
    c = sums.F2KSS(E,L,nnk,2,-1,2,-1,alpha)
    d = sums.F2KSS(E,L,nnk,2,0,2,0,alpha)
    e = sums.F2KSS(E,L,nnk,2,2,2,2,alpha)
    f = sums.F2KSS(E,L,nnk,0,0,2,-2,alpha)
    g = sums.F2KSS(E,L,nnk,0,0,2,0,alpha)
    h = sums.F2KSS(E,L,nnk,2,-2,2,0,alpha)
    i = sums.F2KSS(E,L,nnk,2,-1,2,1,alpha)

    out = np.diag([a,b,c,d,c,e])
    out[0][1] = f; out[1][0] = f
    out[0][3] = g; out[3][0] = g
    out[1][3] = h; out[3][1] = h
    out[2][4] = i; out[4][2] = i
    return out

  # k=(a,a,a)
  elif nnk[0]==nnk[1]==nnk[2]!=0:
    a = sums.F2KSS(E,L,nnk,0,0,0,0,alpha)
    b = sums.F2KSS(E,L,nnk,2,-2,2,-2,alpha)
    c = sums.F2KSS(E,L,nnk,2,0,2,0,alpha)
    d = sums.F2KSS(E,L,nnk,0,0,2,-2,alpha)
    e = sums.F2KSS(E,L,nnk,2,-2,2,-1,alpha)
    f = sums.F2KSS(E,L,nnk,2,-1,2,0,alpha)

    out = np.array([
      [a,d,d,0,d,0],
      [d,b,e,-2*f,e,0],
      [d,e,b,f,e,-sqrt(3)*f],
      [0,-2*f,f,c,f,0],
      [d,e,e,f,b,sqrt(3)*f],
      [0,0,-sqrt(3)*f,0,sqrt(3)*f,c]
    ])
    return out

  # k=(a,b,0)
  elif myabs(nnk[0])!=myabs(nnk[1]) and nnk[2]==0 and nnk[0]!=0!=nnk[1]:
    a = sums.F2KSS(E,L,nnk,0,0,0,0,alpha)
    b = sums.F2KSS(E,L,nnk,2,-2,2,-2,alpha)
    c = sums.F2KSS(E,L,nnk,2,-1,2,-1,alpha)
    d = sums.F2KSS(E,L,nnk,2,0,2,0,alpha)
    e = sums.F2KSS(E,L,nnk,2,1,2,1,alpha)
    f = sums.F2KSS(E,L,nnk,2,2,2,2,alpha)
    g = sums.F2KSS(E,L,nnk,0,0,2,-2,alpha)
    h = sums.F2KSS(E,L,nnk,0,0,2,0,alpha)
    i = sums.F2KSS(E,L,nnk,0,0,2,2,alpha)
    j = sums.F2KSS(E,L,nnk,2,-2,2,0,alpha)
    k = sums.F2KSS(E,L,nnk,2,-2,2,2,alpha)
    l = sums.F2KSS(E,L,nnk,2,-1,2,1,alpha)
    m = sums.F2KSS(E,L,nnk,2,0,2,2,alpha)

    out = np.array([
      [a,g,0,h,0,i],
      [g,b,0,j,0,k],
      [0,0,c,0,l,0],
      [h,j,0,d,0,m],
      [0,0,l,0,e,0],
      [i,k,0,m,0,f]
    ])
    return out

  # k=(a,a,b)
  elif nnk[0]==nnk[1] and myabs(nnk[0])!=myabs(nnk[2]) and nnk[0]!=0!=nnk[2]:
    a = sums.F2KSS(E,L,nnk,0,0,0,0,alpha)
    b = sums.F2KSS(E,L,nnk,2,-2,2,-2,alpha)
    c = sums.F2KSS(E,L,nnk,2,-1,2,-1,alpha)
    d = sums.F2KSS(E,L,nnk,2,0,2,0,alpha)
    e = sums.F2KSS(E,L,nnk,2,2,2,2,alpha)
    f = sums.F2KSS(E,L,nnk,0,0,2,-2,alpha)
    g = sums.F2KSS(E,L,nnk,0,0,2,-1,alpha)
    h = sums.F2KSS(E,L,nnk,0,0,2,0,alpha)
    i = sums.F2KSS(E,L,nnk,2,-2,2,-1,alpha)
    j = sums.F2KSS(E,L,nnk,2,-2,2,0,alpha)
    k = sums.F2KSS(E,L,nnk,2,-1,2,0,alpha)
    l = sums.F2KSS(E,L,nnk,2,-1,2,1,alpha)
    m = sums.F2KSS(E,L,nnk,2,-1,2,2,alpha)

    out = np.array([
      [a,f,g,h,g,0],
      [f,b,i,j,i,0],
      [g,i,c,k,l,m],
      [h,j,k,d,k,0],
      [g,i,l,k,c,-m],
      [0,0,m,0,-m,e]
    ])
    return out

  # k=(a,b,c)
  elif myabs(nnk[0])!=myabs(nnk[1])!=myabs(nnk[2])!=myabs(nnk[0]) and nnk[0]!=0!=nnk[1] and nnk[2]!=0:

    out = np.zeros((6,6))
    for i1 in range(6):
      if i1==0:
        l1=0; m1=0
      else:
        l1=2; m1=i1-3
      for i2 in range(i1,6):
        if i2==0:
          l2=0; m2=0
        else:
          l2=2; m2=i2-3
        out[i1][i2] = sums.F2KSS(E,L,nnk,l1,m1,l2,m2,alpha)
        if i1!=i2:
          out[i2][i1] = out[i1][i2]

    return out

  else:
    print('Not a special nnk')


# Compute all Fmat_k's in a given orbit
def Fmat_shell(E,L,nnk,alpha):
  # k=(0,0,0)
  if list(nnk)==[0,0,0]:
    F_000 = Fmat_k(E,L,nnk,alpha)
    return [F_000]

  # k=(0,0,a)
  elif nnk[0]==nnk[1]==0<nnk[2]:
    U132 = Dmat([1,3,2])
    U321 = Dmat([3,2,1])

    F_00a = Fmat_k(E,L,nnk,alpha)
    F_0a0 = U132 @ F_00a @ U132.T
    F_a00 = U321 @ F_00a @ U321.T

    F_list = list(chop( [F_00a,F_0a0,F_a00] ))
    return F_list*2  # duplicate for Z2 symmetry

  # k=(a,a,0)
  elif nnk[0]==nnk[1]>0==nnk[2]:
    U132 = Dmat([1,3,2])
    U321 = Dmat([3,2,1])
    U1m23 = Dmat([1,-2,3])

    F_aa0 = Fmat_k(E,L,nnk,alpha)
    F_a0a = U132 @ F_aa0 @ U132.T
    F_0aa = U321 @ F_aa0 @ U321.T

    F_ama0 = U1m23 @ F_aa0 @ U1m23.T
    F_a0ma = U132 @ F_ama0 @ U132.T
    F_0ama = U321 @ F_ama0 @ U321.T

    F_list = list(chop(
      [F_aa0,F_a0a,F_0aa, F_ama0,F_a0ma,F_0ama]
      ))

    return F_list*2 # duplicate for Z2 symmetry

  # k=(a,a,a)
  elif nnk[0]==nnk[1]==nnk[2]>0:
    U12m3 = Dmat([1,2,-3])
    U132 = Dmat([1,3,2])
    U321 = Dmat([3,2,1])

    F_aaa = Fmat_k(E,L,nnk,alpha)
    F_aama = U12m3 @ F_aaa @ U12m3.T
    F_amaa = U132 @ F_aama @ U132.T
    F_maaa = U321 @ F_aama @ U321.T

    F_list = list(chop( [F_aaa,F_aama,F_amaa,F_maaa] ))

    return F_list*2 # duplicate for Z2 symmetry

  # k=(a,b,0)
  elif nnk[2]==0<nnk[0]<nnk[1]:
    U213 = Dmat([2,1,3])
    U132 = Dmat([1,3,2])
    U321 = Dmat([3,2,1])
    U1m23 = Dmat([1,-2,3])

    F_ab0 = Fmat_k(E,L,nnk,alpha)
    F_ba0 = U213 @ F_ab0 @ U213.T

    F_a0b = U132 @ F_ab0 @ U132.T
    F_b0a = U321 @ F_a0b @ U321.T

    F_0ab = U213 @ F_a0b @ U213.T
    F_0ba = U132 @ F_0ab @ U132.T


    F_amb0 = U1m23 @ F_ab0 @ U1m23.T
    F_mba0 = U213 @ F_amb0 @ U213.T

    F_a0mb = U132 @ F_amb0 @ U132.T
    F_mb0a = U321 @ F_a0mb @ U321.T

    F_0amb = U213 @ F_a0mb @ U213.T
    F_0mba = U132 @ F_0amb @ U132.T

    F_list = list(chop(
      [F_ab0,F_ba0, F_a0b,F_b0a, F_0ab,F_0ba,  F_amb0,F_mba0, F_a0mb,F_mb0a, F_0amb,F_0mba]
      ))

    return F_list*2 # duplicate for Z2 symmetry

  # k=(a,a,b)
  elif 0<nnk[0]==nnk[1]!=nnk[2]>0:
    U132 = Dmat([1,3,2])
    U321 = Dmat([3,2,1])
    U12m3 = Dmat([1,2,-3])
    U1m23 = Dmat([1,-2,3])
    U213 = Dmat([2,1,3])

    F_aab = Fmat_k(E,L,nnk,alpha)
    F_aba = U132 @ F_aab @ U132.T
    F_baa = U321 @ F_aab @ U321.T

    F_aamb = U12m3 @ F_aab @ U12m3.T
    F_amba = U132 @ F_aamb @ U132.T
    F_mbaa = U321 @ F_aamb @ U321.T


    F_amab = U1m23 @ F_aab @ U1m23.T
    F_maab = U213 @ F_amab @ U213.T

    F_abma = U132 @ F_amab @ U132.T
    F_maba = U321 @ F_abma @ U321.T

    F_bama = U213 @ F_abma @ U213.T
    F_bmaa = U132 @ F_bama @ U132.T

    F_list = list(chop(
      [F_aab,F_aba,F_baa, F_aamb,F_amba,F_mbaa, F_amab,F_abma, F_bama, F_maab, F_maba, F_bmaa]
    ))

    return F_list*2 # duplicate for Z2 symmetry

  # k=(a,b,c)
  elif 0<nnk[0]<nnk[1]<nnk[2]:
    F_abc = Fmat_k(E,L,nnk,alpha)
    F_list = [F_abc]

    p_list = perms_list([1,2,3])
    for p in p_list[1:]:
      p=list(p)
      F_list.append(chop( Dmat(p) @ F_abc @ Dmat(p).T ))

    return F_list*2 # duplicate for Z2 symmetry

  else:
    print('Error in Fmat_shell: Incorrect nnk input')



#######################################################
# Use symmetries to speed up computation of full matrix
def Fmat(E,L,alpha):
  shells = shell_list(E,L)
  #print(shells)
  F_list = []
  for nnk in shells:
    F_list += Fmat_shell(E,L,nnk,alpha)

  # Should probably just return F_list for computations
  return block_diag(*F_list)


# Just compute l'=l=0 portion
@jit(fastmath=True,cache=True)
def Fmat00(E,L,alpha,IPV=0):
  shells = shell_list(E,L)


  
  F_list = []
  for nnk in shells:
    nk = sums.norm(nnk)
    k = nk*2*math.pi/L
    hhk = sums.hh(E, k)
    omk = sqrt(1. + k**2)
    F_list += [(sums.F2KSS(E,L,nnk,0,0,0,0,alpha)+ hhk*IPV/(32*math.pi*2*omk))] * len(shell_nnk_list(nnk))
#  print(F_list)
  return np.diag(F_list)


# Just compute l'=l=2 portion
def Fmat22(E,L,alpha):
  # Right now this doesn't save any time since it computes the full matrix first
  # Not sure if this is worth trying to optimize more
  return l2_proj(Fmat(E,L,alpha))






########################################################################
# Old matrix structure, no symmetries used (slower)
########################################################################

# TB: should let alpha_KSS be an input in all F2 matrices (i.e., not fixed inside)
def full_F2_00_matrix(e,L):

    alpha=0.5

    nklist = list_nnk(e,L)


    F2_00 = []

    for nnk in nklist:

        res = sums.F2KSS(e,L,nnk,0,0,0,0,alpha)
        F2_00.append(res)

    return np.diag(F2_00)



def F2_20_matrix(e,L,m):

    nklist = list_nnk(e,L)

    alpha=0.5

    F2_20 = []
    for nnk in nklist:
        res = sums.F2KSS(e,L,nnk,2,m,0,0,alpha)
        F2_20.append(res)

    return np.diag(F2_20)



def F2_02_matrix(e,L,m):

    nklist = list_nnk(e,L)

    alpha=0.5

    F2_02 = []
    for nnk in nklist:
       # print(nnk)
        res = sums.F2KSS(e,L,nnk,0,0,2,m,alpha)
        F2_02.append(res)

    return np.diag(F2_02)


def F2_22_matrix(e,L,m1,m2):

    alpha=0.5
    nklist = list_nnk(e,L)

    F2_22 = []
    for nnk in nklist:
        #print(nnk)
        res=sums.F2KSS(e,L,nnk,2,m1,2,m2,alpha)
        F2_22.append(res)

    return np.diag(F2_22)


def full_F2_20_matrix(e,L):

    F2_20 = F2_20_matrix(e,L,-2)
    F2_20 = np.vstack((F2_20, F2_20_matrix(e,L,-1)))
    F2_20 = np.vstack((F2_20, F2_20_matrix(e,L,0)))
    F2_20 = np.vstack((F2_20, F2_20_matrix(e,L,1)))
    F2_20 = np.vstack((F2_20, F2_20_matrix(e,L,2)))
    return F2_20


def full_F2_02_matrix(e,L):

    F2_02 = F2_02_matrix(e,L,-2)
    F2_02 = np.hstack((F2_02, F2_02_matrix(e,L,-1)))
    F2_02 = np.hstack((F2_02, F2_02_matrix(e,L,0)))
    F2_02 = np.hstack((F2_02, F2_02_matrix(e,L,1)))
    F2_02 = np.hstack((F2_02, F2_02_matrix(e,L,2)))
    return F2_02


# TB: edited this like I did for Gmatrix.G22
def full_F2_22_matrix(e,L):

    F2_22_m2 = F2_22_matrix(e,L,-2,-2)
    F2_22_m2 = np.hstack((F2_22_m2, F2_22_matrix(e,L,-2,-1)))
    F2_22_m2 = np.hstack((F2_22_m2, F2_22_matrix(e,L,-2,0)))
    F2_22_m2 = np.hstack((F2_22_m2, F2_22_matrix(e,L,-2,1)))
    F2_22_m2 = np.hstack((F2_22_m2, F2_22_matrix(e,L,-2,2)))

    F2_22_m1 = F2_22_matrix(e,L,-1,-2)
    F2_22_m1 = np.hstack((F2_22_m1, F2_22_matrix(e,L,-1,-1)))
    F2_22_m1 = np.hstack((F2_22_m1, F2_22_matrix(e,L,-1,0)))
    F2_22_m1 = np.hstack((F2_22_m1, F2_22_matrix(e,L,-1,1)))
    F2_22_m1 = np.hstack((F2_22_m1, F2_22_matrix(e,L,-1,2)))

    F2_22_0 = F2_22_matrix(e,L,0,-2)
    F2_22_0 = np.hstack((F2_22_0, F2_22_matrix(e,L,0,-1)))
    F2_22_0 = np.hstack((F2_22_0, F2_22_matrix(e,L,0,0)))
    F2_22_0 = np.hstack((F2_22_0, F2_22_matrix(e,L,0,1)))
    F2_22_0 = np.hstack((F2_22_0, F2_22_matrix(e,L,0,2)))

    F2_22_p1 = F2_22_matrix(e,L,1,-2)
    F2_22_p1 = np.hstack((F2_22_p1, F2_22_matrix(e,L,1,-1)))
    F2_22_p1 = np.hstack((F2_22_p1, F2_22_matrix(e,L,1,0)))
    F2_22_p1 = np.hstack((F2_22_p1, F2_22_matrix(e,L,1,1)))
    F2_22_p1 = np.hstack((F2_22_p1, F2_22_matrix(e,L,1,2)))

    F2_22_p2 = F2_22_matrix(e,L,2,-2)
    F2_22_p2 = np.hstack((F2_22_p2, F2_22_matrix(e,L,2,-1)))
    F2_22_p2 = np.hstack((F2_22_p2, F2_22_matrix(e,L,2,0)))
    F2_22_p2 = np.hstack((F2_22_p2, F2_22_matrix(e,L,2,1)))
    F2_22_p2 = np.hstack((F2_22_p2, F2_22_matrix(e,L,2,2)))


    F2_22 = np.vstack((F2_22_m2, F2_22_m1, F2_22_0, F2_22_p1, F2_22_p2))

    return F2_22

# full Ftilde matrix (should let alphaKSS be input)
def Fmat_old(E,L):
  F00 = full_F2_00_matrix(E,L)
  F20 = full_F2_20_matrix(E,L)
  F02 = full_F2_02_matrix(E,L)
  F22 = full_F2_22_matrix(E,L)
  return full_matrix( F00, F20, F02, F22 )
