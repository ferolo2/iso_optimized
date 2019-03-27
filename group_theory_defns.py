import numpy as np
pi=np.pi; conj=np.conjugate; LA=np.linalg
from itertools import permutations as perms
import defns

from numba import jit 
@jit(nopython=True,fastmath=True,cache=True)
def sqrt(x):
    return np.sqrt(x)


####################################################################################
# Group theory definitions
####################################################################################

############################################################
# Basic transformation functions
############################################################
# Find [i,j,k] s.t. R_{ijk}p2 = p1
def Rvec(p2,p1):
  abs_p1 = [abs(x) for x in p1]
  abs_p2 = [abs(x) for x in p2]
  i0 = abs_p1.index(abs_p2[0])
  i1 = abs_p1.index(abs_p2[1])
  if i1==i0:
    i1 = abs_p1.index(abs_p2[1],i0+1,3)
  i2 = [i for i in range(3) if i not in (i0,i1)][0]
  if abs_p1[i2] != abs_p2[2]:
    print('Error')

  ivec = [i0,i1,i2]
  R=[]
  for j in range(3):
    sgn = 1
    if np.sign(p2[j]) != np.sign(p1[ivec[j]]):
      sgn = -1
    R.append(sgn*(ivec[j]+1))
  return R

# Compute p=[i,j,k] = ...p2*p1 s.t. R_p = ...R_{p2}R_{p1}
def R_prod(*argv):
  p = [1,2,3]
  for j in range(len(argv)-1,-1,-1):
    p2 = argv[j]
    p = [np.sign(i)*p[abs(i)-1] for i in p2]
  return p


############################################################
# Permutation matrices S(R)
############################################################
def Smat(R,shell):
  R = list(R); shell = list(shell)
  nnk_list = defns.shell_nnk_list(shell)
  nnk_list = [list(nnk) for nnk in nnk_list]
  Nk = len(nnk_list)

  S = np.zeros((Nk,Nk))
  for i in range(Nk):
    k = nnk_list[i]
    kR = [ np.sign(R[j]) * k[abs(R[j])-1] for j in range(3) ]
    j = nnk_list.index(kR)

    S[i][j] = 1
  return S


############################################################
# Wigner D-matrices D(R) in real Ylm basis
############################################################
def Dmat(R):
  R = list(R)
  if tuple(abs(i) for i in R) not in list(perms([1,2,3])):
    print('Error in Dmat: invalid input')

  # Trivial transformation:
  if R==[1,2,3] or R==[-1,-2,-3]:
    return np.identity(6)

  # Single permutation:
  elif R==[2,1,3] or R==[-2,-1,-3]:
    U = np.identity(6)
    U[2][2]=0; U[2][4]=1
    U[4][2]=1; U[4][4]=0
    U[5][5]=-1
    return U

  elif R==[1,3,2] or R==[-1,-3,-2]:
    U = np.zeros((6,6))
    U[0][0]=1
    U[1][4]=1; U[2][2]=1; U[4][1]=1
    U[3][3] = -1/2;       U[3][5] = -sqrt(3)/2
    U[5][3] = -sqrt(3)/2; U[5][5] = 1/2
    return U

  elif R==[3,2,1] or R==[-3,-2,-1]:
    U = np.zeros((6,6))
    U[0][0]=1
    U[1][2]=1; U[2][1]=1; U[4][4]=1
    U[3][3] = -1/2;       U[3][5] = sqrt(3)/2
    U[5][3] = sqrt(3)/2; U[5][5] = 1/2
    return U

  # Cyclic permution:
  elif R==[2,3,1] or R==[-2,-3,-1]:
    return defns.chop( Dmat([1,3,2]) @ Dmat([2,1,3]) )
  
  elif R==[3,1,2] or R==[-3,-1,-2]:
    return defns.chop( Dmat([3,2,1]) @ Dmat([2,1,3]) )
  
  # Single negation
  elif R==[1,2,-3] or R==[-1,-2,3]:
    return defns.chop( np.diag([1,1,-1,1,-1,1]) )
  
  elif R[0]*R[1]>0 and R[0]*R[2]<0:
    return defns.chop( Dmat([1,2,-3]) @ Dmat([R[0],R[1],-R[2]]) )
  
  elif R[0]*R[2]>0 and R[0]*R[1]<0:
    return defns.chop( Dmat([1,3,2]) @ Dmat([R[0],R[2],R[1]]) )
  
  elif R[1]*R[2]>0 and R[0]*R[1]<0:
    return defns.chop( Dmat([3,2,1]) @ Dmat([R[2],R[1],R[0]]) )

  else:
    print('Error in Dmat: This should never trigger')


############################################################
# Cubic group, little groups
############################################################
# Create list of all 48 permutations w/ any # of negations (i.e., full cubic group w/ inversions Oh)
def Oh_list():
  Oh_list = list(perms([1,2,3]))
  Oh_list += list(perms([1,2,-3]))
  Oh_list += list(perms([1,-2,3]))
  Oh_list += list(perms([-1,2,3]))
  Oh_list += list(perms([1,-2,-3]))
  Oh_list += list(perms([-1,2,-3]))
  Oh_list += list(perms([-1,-2,3]))
  Oh_list += list(perms([-1,-2,-3]))

  Oh_list = [ list(R) for R in Oh_list ]
  return Oh_list


# Little group for given k
def little_group(shell):
  shell = list(shell)
  # 000
  if shell==[0,0,0]:
    return Oh_list()

  # 00a
  elif shell[0]==shell[1]==0:
    return [[1,2,3],[-1,2,3],[1,-2,3],[-1,-2,3],[2,1,3],[-2,1,3],[2,-1,3],[-2,-1,3]]
  
  # a00 (temporary)
  elif shell[1]==shell[2]==0:
    return [[1,2,3],[1,-2,3],[1,2,-3],[1,-2,-3],[1,3,2],[1,-3,2],[1,3,-2],[1,-3,-2]]

  # aa0
  elif shell[0]==shell[1]!=shell[2]==0:
    return [[1,2,3],[1,2,-3],[2,1,3],[2,1,-3]]

  # aaa
  elif shell[0]==shell[1]==shell[2]:
    return [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

  # ab0
  elif 0!=shell[0]!=shell[1]!=0==shell[2]:
    return [[1,2,3],[1,2,-3]]

  # aab
  elif shell[0]==shell[1]!=shell[2]:
    return [[1,2,3],[2,1,3]]

  # abc
  elif 0!=shell[0]!=shell[1]!=shell[2]!=shell[0]!=0 and shell[1]!=0:
    return [[1,2,3]]
  
  else:
    print('Error: invalid shell input')


###########################################################
# Irreps, conjugacy classes, characters
###########################################################
# List of cubic group irreps
def irrep_list():
  return ['A1+','A2+','E+','T1+','T2+','A1-','A2-','E-','T1-','T2-']

# Dimension of irrep
def irrep_dim(I):
  if I in ['A1+','A1','A2+','A2','A1-','A2-']:
    return 1
  elif I in ['E+','E','E-']:
    return 2
  elif I in ['T1+','T1','T2+','T2','T1-','T2-']:
    return 3
  else:
    print('Error: invalid irrep in irrep_dim')


# Compute conjugacy class of R_p, where p=[i,j,k]
def conj_class(p):
  p = list(p)
  p_abs = [abs(x) for x in p]

  N_negs = sum([1 for x in range(3) if p[x]<0])
  N_correct = sum([1 for x in range(3) if p_abs[x]==x+1])

  if N_correct == 3:
    if N_negs == 0:
      return 'E'
    elif N_negs == 2:
      return 'C4^2'
    
    elif N_negs ==3:
      return 'i'
    elif N_negs == 1:
      return 'sigma_h'    
  
  elif N_correct == 0:
    if (N_negs % 2) == 0:
      return 'C3'
    else:
      return 'S6'
  
  elif N_correct == 1:
    i_correct = [i for i in range(3) if p_abs[i]==i+1][0]
    if (N_negs % 2) == 1:
      if p[i_correct] < 0:
        return 'C2'
      else:
        return 'C4'
    else:
      if p[i_correct] > 0:
        return 'sigma_d'
      else:
        return 'S4'
  else:
    print('Error in conj_class: should never reach here')
  

# Compute character for R_p in irrep I, where p=[i,j,k]
def chi(p,I):
  cc = conj_class(p)

  if I in ['A1+','A1']:
    return 1

  elif I in ['A2+','A2']:
    if cc in ['C2','C4','sigma_d','S4']:
      return -1
    else:
      return 1

  elif I in ['E+','E']:
    if cc in ['E','C4^2','i','sigma_h']:
      return 2
    elif cc in ['C3','S6']:
      return -1
    else:
      return 0
  
  elif I in ['T1+','T1']:
    if cc in ['E','i']:
      return 3
    elif cc in ['C3','S6']:
      return 0
    elif cc in ['C4','S4']:
      return 1
    else:
      return -1
  
  elif I in ['T2+','T2']:
    if cc in ['E','i']:
      return 3
    elif cc in ['C3','S6']:
      return 0
    elif cc in ['C2','sigma_d']:
      return 1
    else:
      return -1
  
  elif I=='A1-':
    if cc in ['E','C3','C4^2','C4','C2']:
      return 1
    else:
      return -1
  
  elif I=='A2-':
    if cc in ['E','C3','C4^2','S4','sigma_d']:
      return 1
    else:
      return -1
  
  elif I=='E-':
    if cc in ['E','C4^2']:
      return 2
    elif cc in ['i','sigma_h']:
      return -2
    elif cc=='S6':
      return 1
    elif cc=='C3':
      return -1
    else:
      return 0
  
  elif I=='T1-':
    if cc=='E':
      return 3
    elif cc=='i':
      return -3
    elif cc in ['C4','sigma_h','sigma_d']:
      return 1
    elif cc in ['C2','C4^2','S4']:
      return -1
    else:
      return 0

  elif I=='T2-':
    if cc=='E':
      return 3
    elif cc=='i':
      return -3
    elif cc in ['C2','S4','sigma_h']:
      return 1
    elif cc in ['C4','C4^2','sigma_d']:
      return -1    
    else:
      return 0


###########################################################
# Irrep projection subspaces/eigenvalue decomposition
###########################################################
# Dimension of irrep projection subspace for given shell
def subspace_dim_o(shell,I):
  s = 0
  for R in little_group(shell):
    s += chi(R,I)*np.trace(Dmat(R))
  return int(s/len(little_group(shell)) * irrep_dim(I))

# Dimension of irrep projection subspace for given shell & l
def subspace_dim_o_l(shell,I,l):
  if l==0:
    s = 0
    for R in little_group(shell):
      s += chi(R,I)
    return int(s/len(little_group(shell)) * irrep_dim(I))
  elif l==2:
    return subspace_dim_o(shell,I) - subspace_dim_o_l(shell,I,0)



###########################################################
# Graveyard (old code)
###########################################################
# Compute R(n,theta) for any axis n and angle theta
def Rmat(n,t):
  N = LA.norm(n)
  n = [i/N for i in n]
  n_sin = np.array([
    [0,-n[2],n[1]],
    [n[2],0,-n[0]],
    [-n[1],n[0],0]
    ])
  
  return defns.chop(np.cos(t)*np.identity(3) + np.sin(t)*n_sin + (1-np.cos(t))*np.outer(n,n))
