import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg

import F2_alt, Gmatrix, sums_alt as sums, K2i_mat
from defns import chop, full_matrix
from numba import jit
#################################################################
# Compute full matrix H = 1/(2*omega*K2) + Ftilde + Gtilde
#################################################################

# Individual matrix element
def H(E,L,npvec,lp,mp,nkvec,l,m,a0,r0,P0,a2,alpha):
  
  if nkvec==npvec:    
    
    F2term = sums.F2KSS(E,L,nkvec,lp,mp,l,m,alpha)
    
    if l==lp and m==mp:
      kvec = [i*2*pi/L for i in nkvec]
      K2term = K2i_mat.K2inv(E,kvec,l,m,a0,r0,P0,a2)
    else:
      K2term = 0
  else:
    F2term = 0
    K2term = 0

  # may have (lp,mp) and (l,m) reversed in F2KSS
  return K2term + F2term + Gmatrix.G(E,L,npvec,nkvec,lp,mp,l,m)


# Construct full H matrix (new structure, uses faster Fmat)
def Hmat(E,L,a0,r0,P0,a2,alpha):
  Ft = F2_alt.Fmat(E,L,alpha)
  Gt = Gmatrix.Gmat(E,L)
  K2it = K2i_mat.K2inv_mat(E,L,a0,r0,P0,a2)

  return chop(K2it + Ft + Gt)


# Just compute l'=l=0 portion of H
@jit(fastmath=True,cache=True)
def Hmat00(E,L,a0,r0,P0,alpha,IPV=0):
  Ft00 = F2_alt.Fmat00(E,L,alpha,IPV)
  Gt00 = Gmatrix.Gmat00(E,L)
  K2it00 = K2i_mat.K2inv_mat00(E,L,a0,r0,P0,IPV)

  return chop(K2it00 + Ft00 + Gt00)


# Just compute l'=l=2 portion of H
def Hmat22(E,L,a2,alpha):
  Ft22 = F2_alt.Fmat22(E,L,alpha)
  Gt22 = Gmatrix.Gmat22(E,L)
  K2it22 = K2i_mat.K2inv_mat22(E,L,a2)

  return chop(K2it22 + Ft22 + Gt22)

#################################################################
# Old matrix structure (uses slower Fmat_old)
#################################################################
# should let alpha be an input for F2 matrices
def Hmat_old(E,L,a0,r0,P0,a2):
  Hmat00 = K2i_mat.K2inv_mat0(E,L,a0,r0,P0) + F2_alt.full_F2_00_matrix(E,L) + Gmatrix.G00(E,L)
  Hmat20 = F2_alt.full_F2_20_matrix(E,L) + Gmatrix.G20(E,L)
  Hmat02 = F2_alt.full_F2_02_matrix(E,L) + Gmatrix.G02(E,L)
  Hmat22 = K2i_mat.K2inv_mat2(E,L,a2) + F2_alt.full_F2_22_matrix(E,L) + Gmatrix.G22(E,L)

  return full_matrix( Hmat00, Hmat20, Hmat02, Hmat22 )


# H^-1 calculation using (l',l) blocks
def Hmatinv_block(E,L,a0,r0,P0,a2):
  Ai = LA.inv( K2i_mat.K2inv_mat0(E,L,a0,r0,P0) + F2_alt.full_F2_00_matrix(E,L) + Gmatrix.G00(E,L) )
  B = F2_alt.full_F2_02_matrix(E,L) + Gmatrix.G02(E,L)
  C = F2_alt.full_F2_20_matrix(E,L) + Gmatrix.G20(E,L)
  D = K2i_mat.K2inv_mat2(E,L,a2) + F2_alt.full_F2_22_matrix(E,L) + Gmatrix.G22(E,L)

  Ei = LA.inv( D-C*Ai*B )

  out00 = Ai + Ai@B@Ei@C@Ai
  out20 = -Ei@C@Ai
  out02 = -Ai@B@Ei
  out22 = Ei

  return full_matrix( out00, out20, out02, out22 )


  
