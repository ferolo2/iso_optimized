import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg
from scipy.linalg import block_diag

import F2_alt, Gmatrix, sums_alt as sums
import defns
#from defns import omega, E2k, qst, list_nnk, lm_idx, full_matrix
from numba import jit,njit

# Calculate K2inverse (really K2inverse/(2*omega))
@njit(fastmath=True,cache=True)
def K2inv(E,kvec,l,m,a0,r0,P0,a2,IPV=0):
  k = sums.norm(kvec)
  omk = defns.omega(k)
  E2star = defns.E2k(E,k)
#  qk = defns.qst(E,k)
  qk2 = defns.qst2(E,k)
  h = sums.hh(E,k)

  if h==0:
    print('Error: invalid kvec in K2inv')

  if l==m==0:
    out = 1/(32*pi*omk*E2star) * ( -1/a0 + r0*qk2/2 + P0*r0**3*qk2**2 + np.sqrt(np.sqrt(qk2**2))*(1-h)) - h*IPV/(32*pi*2*omk)

  elif l==2 and -2<=m<=2:
    #out = 1/(32*pi*omk*E2star*qk**4) * ( -1/a2**5 + abs(qk)**5*(1-h) )
    out = 1/(32*pi*omk*E2star) * ( -1/a2**5 + abs( np.sqrt(np.sqrt(qk2**2))  )**5*(1-h) ) # TB, no q

  else:
    return 0

  if out.imag > 1e-15:
    print('Error in K2inv: imaginary part in output')
  else:
    out = out.real
  return out

def K2inv_old(E,kvec,l,m,a0,r0,P0,a2,IPV=0.):
  k = LA.norm(kvec)
  omk = defns.omega(k)
  E2star = defns.E2k(E,k)
  qk = defns.qst(E,k)
  h = sums.hh(E,k)

  if h==0:
    print('Error: invalid kvec in K2inv')

  if l==m==0:
    out = 1/(32*pi*omk*E2star) * ( -1/a0 + r0*qk**2/2 + P0*r0**3*qk**4 + abs(qk)*(1-h)) - h*IPV/(32*pi*2*omk)

  elif l==2 and -2<=m<=2:
    #out = 1/(32*pi*omk*E2star*qk**4) * ( -1/a2**5 + abs(qk)**5*(1-h) )
    out = 1/(32*pi*omk*E2star) * ( -1/a2**5 + abs(qk)**5*(1-h) ) # TB, no q

  else:
    return 0

  if out.imag > 1e-15:
    print('Error in K2inv: imaginary part in output')
  else:
    out = out.real
  return out




# Make full K2inv matrix (new structure)
def K2inv_mat(E,L,a0,r0,P0,a2):
  nnk_list = defns.list_nnk(E,L)
  N = len(nnk_list)

  K2i_full = []
  for k in range(N):
    kvec = [ ki*2*pi/L for ki in nnk_list[k] ]

    K2i_k_diag = []
    for i in range(6):
      [l,m] = defns.lm_idx(i)
      K2i_k_diag.append(K2inv(E,kvec,l,m,a0,r0,P0,a2))

    K2i_k = np.diag(K2i_k_diag)
    K2i_full.append(K2i_k)

  return block_diag(*K2i_full)


# Just compute l'=l=0 portion of K2inv
#@jit()
def K2inv_mat00(E,L,a0,r0,P0,IPV=0):
  nklist = defns.list_nnk(E,L)
  K2inv00 = []
  for nkvec in nklist:
    kvec = [i*2*pi/L for i in nkvec]
    K2inv00.append(K2inv(E,kvec,0,0,a0,r0,P0,0,IPV))
  return np.diag(K2inv00)


# Just compute l'=l=2 portion of K2inv
def K2inv_mat22(E,L,a2):
  nklist = defns.list_nnk(E,L)
  K2inv22 = []
  for nkvec in nklist:
    kvec = [i*2*pi/L for i in nkvec]
    K2inv22.append(K2inv(E,kvec,2,0,0,0,0,a2)*np.identity(5))
  return block_diag(*K2inv22)


# Just compute l'=l=0 portion of K2inv in A1+ irrep (for H^-1 in iso approx)
def K2inv_mat00_A1(E,L,a0,r0,P0):
  shells = defns.shell_list(E,L)
  out = []
  for nkvec in shells:
    kvec = [i*2*pi/L for i in nkvec]
    out.append(K2inv(E,kvec,0,0,a0,r0,P0,0))
  return np.diag(out)


###########################################################
# Old matrix structure
###########################################################

# # Makes 00 block of K2inv (identical to above function)
# def K2inv_mat0(E,L,a0,r0,P0):
#   nklist = defns.list_nnk(E,L)
#   K2inv0 = []
#   for nkvec in nklist:
#     kvec = [i*2*pi/L for i in nkvec]
#     K2inv0.append(K2inv(E,kvec,0,0,a0,r0,P0,0))
#   return np.diag(K2inv0)

# Makes 22 block of K2inv
def K2inv_mat2(E,L,a2):
  nklist = defns.list_nnk(E,L)

  K2inv2 = []
  for nkvec in nklist:
    kvec = [i*2*pi/L for i in nkvec]
    K2inv2.append(K2inv(E,kvec,2,0,0,0,0,a2))

  # Potentially faster method (pre-allocates memory)
  # blocksize = len(nklist)
  # K2inv2 = [None]*blocksize
  # for i in range(0,blocksize):
  #   kvec = [j*2*pi/L for j in nklist[i]]
  #   K2inv2[i] = K2inv(E,kvec,2,0,0,0,0,a2)

  K2_block = np.diag(K2inv2)

  return np.kron(np.identity(5),K2_block)

# Note: 20 and 02 blocks of K2inv are zero
