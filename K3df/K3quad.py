import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg;

from K3A import K3A; from K3B import K3B
from K3cubicA import K3cubicA; from K3cubicB import K3cubicB
from defns import list_nnk, lm_idx, chop, full_matrix, qst

#################################################################
# Full quadratic-order threshold expansion of K3df
#################################################################

# Note: input order for all functions is (E,outgoing momenta,incoming momenta)


def K3quad(E,pvec,lp,mp,kvec,l,m,K0,K1,K2,A,B,Ytype='r'):
  d = E**2-9

  out = 0
  if lp==mp==l==m==0:
    out += K0 + K1*d + K2*d**2
  if A!=0:
    out += A*K3A(E,pvec,lp,mp,kvec,l,m,Ytype)
  if B!=0:
    out += B*K3B(E,pvec,lp,mp,kvec,l,m,Ytype)

  #qp = qst(E,LA.norm(pvec))
  #qk = qst(E,LA.norm(kvec))
  #out *= qp**lp * qk**l      # q factors are NOT included here (no q)

  if (Ytype=='r' or Ytype=='real') and out.imag>1e-15:
    print('Error in K3quad: imaginary part in real basis output')
  else:
    out = out.real
  return out


# Full matrix (new structure)
def K3mat(E,L,K0,K1,K2,A,B,Ytype='r'):
  nnk_list = list_nnk(E,L)
  N = len(nnk_list)

  K3full = []
  for p in range(N):
    pvec = [ i*2*pi/L for i in nnk_list[p] ]

    K3p = []
    for k in range(N):
      kvec = [ i*2*pi/L for i in nnk_list[k] ]

      K3pk = np.zeros((6,6))
      for i1 in range(6):
        [lp,mp] = lm_idx(i1)
        for i2 in range(6):
          [l,m] = lm_idx(i2)

          K3pk[i1,i2] = K3quad(E,pvec,lp,mp,kvec,l,m,K0,K1,K2,A,B,Ytype)
          #K3pk[i1,i2] += 100*K3cubicA(E,pvec,lp,mp,kvec,l,m) # BLEH temporary
#          K3pk[i1,i2] += 100*K3cubicB(E,pvec,lp,mp,kvec,l,m) # BLEH temporary

      K3p.append(K3pk)

    K3full.append(K3p)

  return chop(np.block(K3full))



#################################################################
# Old matrix structure
def K3mat_old(E,L,K0,K1,K2,A,B,Ytype):
  nnk_list = list_nnk(E,L)
  N = len(nnk_list)

  K3_00 = np.zeros((N,N))
  K3_02 = np.zeros((N,5*N))
  K3_20 = np.zeros((5*N,N))
  K3_22 = np.zeros((5*N,5*N))

  for i in range(N):
    pvec = [ npi*2*pi/L for npi in nnk_list[i] ]
    for j in range(N):
      kvec = [ nki*2*pi/L for nki in nnk_list[j] ]

      K3_00[i][j] = K3quad(E,pvec,0,0,kvec,0,0,K0,K1,K2,A,B,Ytype)

      for mp in range(-2,3):
        I = (mp+2)*N + i
        K3_20[I][j] = K3quad(E,pvec,2,mp,kvec,0,0,K0,K1,K2,A,B,Ytype)

        for m in range(-2,3):
          J = (m+2)*N + j
          K3_22[I][J] = K3quad(E,pvec,2,mp,kvec,2,m,K0,K1,K2,A,B,Ytype)

          if mp==2: # only do this once (mp==2 is arbitrary)
            K3_02[i][J] = K3quad(E,pvec,0,0,kvec,2,m,K0,K1,K2,A,B,Ytype)

  return full_matrix(K3_00,K3_20,K3_02,K3_22)
