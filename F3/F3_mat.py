import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg
from F2_alt import Fmat, Fmat00, Fmat22, Fmat_old
from H_mat import Hmat, Hmat00, Hmat22, Hmat_old
from defns import chop
import projections as proj
from numba import jit,njit
import Gmatrix, K2i_mat
##############################################################
# Compute full matrix F3 = 1/L**3 * (Ft/3 - Ft@Hi@Ft)
# Uses new structure w/ faster Fmat, Hmat
##############################################################

# Compute full F3 matrix
# (symmetries of F2 exploited to speed up computation)
def F3mat(E,L,a0,r0,P0,a2,alpha):
  F = Fmat(E,L,alpha)
  Hi = chop(LA.inv(Hmat(E,L,a0,r0,P0,a2,alpha)))
  return 1/L**3 * chop((1/3*F - F@Hi@F))

# Inverse matrix
def F3i_mat(E,L,a0,r0,P0,a2,alpha):
  return chop(LA.inv(F3mat(E,L,a0,r0,P0,a2,alpha)))

# Inverse matrix eigenvalues, sorted least->greatest
def F3i_eigs(E,L,a0,r0,P0,a2,alpha):
  F3i = F3i_mat(E,L,a0,r0,P0,a2,alpha)
  return np.array(sorted(LA.eigvals(F3i))).real


# Just compute l'=l=0 portion of F3
@jit(fastmath=True,cache=True)
def F3mat00_old(E,L,a0,r0,P0,a2,alpha,IPV=0):
  F00 = Fmat00(E,L,alpha,IPV)
  Hi00 = chop(LA.inv(Hmat00(E,L,a0,r0,P0,alpha,IPV)))
  return 1/L**3 * chop((1/3*F00 - F00@Hi00@F00))

@jit(fastmath=True,cache=True)
def F3mat00(E,L,a0,r0,P0,a2,alpha,IPV=0):
  F00 = Fmat00(E,L,alpha,IPV)  
  Gt00 = Gmatrix.Gmat00(E,L)
  K2it00 = K2i_mat.K2inv_mat00(E,L,a0,r0,P0,IPV)  
  Hi00 = chop(LA.inv( K2it00 + F00 + Gt00  ))
  return 1/L**3 * chop((1/3*F00 - F00@Hi00@F00))

@njit(fastmath=True,cache=True)
def LAinv(x):
  return LA.inv(x)


@jit(fastmath=True,cache=True)
def F3mat00iso(E,L,Lista0,r0,P0,a2,alpha,IPV=0):
  F00 = Fmat00(E,L,alpha,IPV)
  F00o3 = 1./3*F00
  Gt00 = Gmatrix.Gmat00(E,L)
  res = []
  ones = np.ones(len(F00))
  for a0 in Lista0:
    K2it00 = K2i_mat.K2inv_mat00(E,L,a0,r0,P0,IPV)
    Hi00 = chop(LAinv( K2it00 + F00 + Gt00  ))
    f3mat = 1/L**3 * chop((F00o3 - F00@Hi00@F00))
    res.append(1./(ones@f3mat@ones))
  return res
    

def F3imat00(E,L,a0,r0,P0,a2,alpha,IPV=0):
  return chop(LA.inv(F3mat00(E,L,a0,r0,P0,a2,alpha,IPV)))


# Just compute l'=l=2 portion of F3
def F3mat22(E,L,a0,r0,P0,a2,alpha):
  F22 = Fmat22(E,L,alpha)
  Hi22 = chop(LA.inv(Hmat22(E,L,a2,alpha)))
  return 1/L**3 * chop((1/3*F22 - F22@Hi22@F22))

def F3imat22(E,L,a0,r0,P0,a2,alpha):
  return chop(LA.inv(F3mat22(E,L,a0,r0,P0,a2,alpha)))


#######################################################
# Projection onto irrep I (l=0 and l=2 contributions)

# F3, irrep I projection
def F3_I_mat(E,L,a0,r0,P0,a2,alpha,I):
  return proj.irrep_proj(F3mat(E,L,a0,r0,P0,a2,alpha),E,L,I)


# Inverse matrix, irrep I projection
def F3i_I_mat(E,L,a0,r0,P0,a2,alpha,I):
  F3_I = F3_I_mat(E,L,a0,r0,P0,a2,alpha,I)

  if F3_I.shape==():
    return 1/F3_I
  else:
    return chop(LA.inv(F3_I))

# Inverse matrix eigenvalues (irrep I projection)
def F3i_I_eigs(E,L,a0,r0,P0,a2,alpha,I):
  F3i_I = F3i_I_mat(E,L,a0,r0,P0,a2,alpha,I)
  return np.array(sorted(LA.eigvals(F3i_I))).real


# l'=l=0 inverse matrix eigenvalues (irrep I projection)
def F3i_00_I_eigs(E,L,a0,r0,P0,a2,alpha,I):
  F3 = F3mat00(E,L,a0,r0,P0,a2,alpha)
  F3i = chop(LA.inv(F3))
  F3i_I = proj.irrep_proj_00(F3i,E,L,I)
  return np.array(sorted(LA.eigvals(F3i_I))).real

# l'=l=2 inverse matrix eigenvalues (irrep I projection)
def F3i_22_I_eigs(E,L,a0,r0,P0,a2,alpha,I):
  F3 = F3mat22(E,L,a0,r0,P0,a2,alpha)
  F3i = chop(LA.inv(F3))
  F3i_I = proj.irrep_proj_22(F3i,E,L,I)
  return np.array(sorted(LA.eigvals(F3i_I))).real


#######################################################
# Isotropic approx (A1+ projection, l=0)
# These functions correctly use only the l=l'=0 part of each matrix appearing in F3

# F3, isotropic approx
def F3_iso_mat(E,L,a0,r0,P0,a2,alpha):
  return proj.iso_proj00(F3mat00(E,L,a0,r0,P0,a2,alpha),E,L)

# Inverse matrix, isotropic approx
def F3i_iso_mat(E,L,a0,r0,P0,a2,alpha):
  F3_iso = F3_iso_mat(E,L,a0,r0,P0,a2,alpha)

  if F3_iso.shape==():
    return 1/F3_iso
  else:
    return chop(LA.inv(F3_iso))

# Inverse matrix eigenvalues (isotropic approx)
def F3i_iso_eigs(E,L,a0,r0,P0,a2,alpha):
  F3i_iso = F3i_iso_mat(E,L,a0,r0,P0,a2,alpha)
  return np.array(sorted(LA.eigvals(F3i_iso))).real



################################################################
# Graveyard (old code)
################################################################
# Single shell projection (not useful since F3 connects different shells & \ell's)

# Inverse matrix eigenvalues (irrep I projection, single shell & l)
def F3i_I_eigs_o_l(E,L,a0,r0,P0,a2,alpha,I,shell,l):
  F3 = F3mat(E,L,a0,r0,P0,a2,alpha)
  
  F3_I_o_l = proj.irrep_proj_o_l( F3,E,L,I,shell,l )
  F3i_I_o_l = chop(LA.inv(F3_I_o_l))

  # F3i = chop(LA.inv(F3)) # temporary BLEH
  # F3i_I_o_l = proj.irrep_proj_o_l( F3i,E,L,I,shell,l ) # temporary BLEH
  return np.array(sorted(LA.eigvals(F3i_I_o_l))).real


# Inverse matrix eigenvalues (irrep I projection, single shell, contains l=0 and l=2)
def F3i_I_eigs_o(E,L,a0,r0,P0,a2,alpha,I,shell):
  F3 = F3mat(E,L,a0,r0,P0,a2,alpha)
  
  F3_I_o = proj.irrep_proj_o( F3,E,L,I,shell )
  F3i_I_o = chop(LA.inv(F3_I_o))

  # F3i = chop(LA.inv(F3)) # temporary BLEH
  # F3i_I_o = proj.irrep_proj_o( F3i,E,L,I,shell ) # temporary BLEH
  return np.array(sorted(LA.eigvals(F3i_I_o))).real


#######################################################
# A1+ projection (redundant due to F3i_I functions)

# F3, A1 projection
def F3_A1_mat(E,L,a0,r0,P0,a2,alpha):
  return proj.A1_proj(F3mat(E,L,a0,r0,P0,a2,alpha),E,L)


# Inverse matrix, A1 projection
def F3i_A1_mat(E,L,a0,r0,P0,a2,alpha):
  F3_A1 = F3_A1_mat(E,L,a0,r0,P0,a2,alpha)

  if F3_A1.shape==():
    return 1/F3_A1
  else:
    return chop(LA.inv(F3_A1))

# Inverse matrix eigenvalues (A1 projection)
def F3i_A1_eigs(E,L,a0,r0,P0,a2,alpha):
  F3i_A1 = F3i_A1_mat(E,L,a0,r0,P0,a2,alpha)
  return np.array(sorted(LA.eigvals(F3i_A1))).real


#######################################################
# Old matrix structure (slower)

# Compute full F3 matrix
# TB: should add alpha as input for F2 matrices
def F3mat_old(E,L,a0,r0,P0,a2):

  F = Fmat_old(E,L)

  Hi = LA.inv(Hmat_old(E,L,a0,r0,P0,a2))
  #Hi = H_mat.Hmatinvblock(E,L,a0,r0,P0,a2)

  return 1/L**3 * ( 1/3*F - F@Hi@F )

def F3i_mat_old(E,L,a0,r0,P0,a2):
  return LA.inv(F3mat_old(E,L,a0,r0,P0,a2))



