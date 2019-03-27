from defns import omega, E2k, qst, y2,y2real
import numpy as np
from numba import jit
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg;
# 1j = imaginary number

#################################################################
# Want to compute K3B = sum(t_{ij}**2)
# First find p2sum = sum((pi.pj)**2) = wss + wds + wsd + wdd, and break up
# wdd = wddpp + wddmp + wddpm + wddmm (see notes)
#################################################################

# Note: input order for all functions is (E,outgoing momenta,incoming momenta)

@jit(nopython=True, parallel=True,fastmath=True) #FRL this speeds up like 20%                                                                                                           
def npsqrt(x):
    return np.sqrt(x)

@jit(nopython=True, parallel=True,fastmath=True) #FRL this speeds up like 20%                                                                                                           
def norm(nnk):
    nk=0.
    for i in nnk:
        nk += i**2
    return npsqrt(nk)

@jit(nopython=True,parallel=True,fastmath=True)
def mydot(x,y):
    res = 0.
    for i in range(len(x)):
        res+=x[i]*y[i]
    return res



#################################################################
# wss and wddpp (l=lp=0 terms)
#################################################################
@jit(fastmath=True,cache=True)
def wss(E,pvec,lp,mp,kvec,l,m):
  if l==m==lp==mp==0:
    k=norm(kvec); p=norm(pvec)
    return ( omega(k)*omega(p) - mydot(kvec,pvec) )**2
  else:
    return 0
@jit(fastmath=True,cache=True)
def wddpp(E,pvec,lp,mp,kvec,l,m):
  if l==m==lp==mp==0:
    k=norm(kvec); p=norm(pvec)
    return 1/4*( (E-omega(k))*(E-omega(p)) - mydot(kvec,pvec) )**2
  else:
    return 0

#################################################################
# wds and wsd (l=2,lp=0; l=0,lp=2; and l=lp=0 terms)
#################################################################
#@jit(fastmath=True,cache=True)
def pstark(E,pvec,kvec): # (see notes)
  k=norm(kvec); p=norm(pvec)
  beta_k = k/(E-omega(k)); gam_k = sqrt(1/(1-beta_k**2))
  khat = [ki/k for ki in kvec] if k!=0 else kvec

  return [pvec[i] + ( gam_k*beta_k*omega(p) + (gam_k-1)*mydot(pvec,khat) ) * khat[i] for i in range(0,3)]
@jit(fastmath=True,cache=True)
def wds(E,pvec,lp,mp,kvec,l,m,Ytype='r'):
  if lp==mp==0:
    k=norm(kvec); p=norm(pvec)
    psk = pstark(E,pvec,kvec)
    if l==m==0:
      return 1/2*(E*omega(p)-sqrt(wss(E,pvec,0,0,kvec,0,0)))**2 + 2/3*(qst(E,k)*norm(psk))**2
    elif l==2:
      #return 4/15 * qst(E,k)**2 * conj(y2(psk,m,Ytype))
#      return 4/15 * conj(y2(psk,m,Ytype)) # TB, no q
      return 4/15 * conj(y2real(psk,m)) # TB, no q
  else:
    return 0
  
@jit(fastmath=True,cache=True)
def wsd(E,pvec,lp,mp,kvec,l,m,Ytype):
  return conj(wds(E,kvec,l,m,pvec,lp,mp,Ytype))

#################################################################
# wddmp and wddpm (l=2,lp=0; l=0,lp=2; and l=lp=0 terms)
#################################################################
def p12stark(E,pvec,kvec): # (see notes)
  k=norm(kvec); p=norm(pvec)
  beta_k = k/(E-omega(k)); gam_k = sqrt(1/(1-beta_k**2))
  khat = [ki/k for ki in kvec] if k!=0 else kvec

  return [-pvec[i] + ( gam_k*beta_k*(E-omega(p)) - (gam_k-1)*mydot(pvec,khat) ) * khat[i] for i in range(0,3)]

@jit(fastmath=True,cache=True)
def wddmp(E,pvec,lp,mp,kvec,l,m,Ytype='r'):
  if lp==mp==0:
    k=norm(kvec)
    p12sk = p12stark(E,pvec,kvec)
    if l==m==0:
      return 1/3*(qst(E,k)*norm(p12sk))**2
    elif l==2:
      #return 2/15 * qst(E,k)**2 * conj(y2(p12sk,m,Ytype))
#      return 2/15 * conj(y2(p12sk,m,Ytype)) # TB, no q
      return 2/15 * conj(y2real(p12sk,m)) # TB, no q

  else:
    return 0

@jit(fastmath=True,cache=True)
def wddpm(E,pvec,lp,mp,kvec,l,m,Ytype):
  return conj(wddmp(E,kvec,l,m,pvec,lp,mp,Ytype))

#################################################################
# wddmm (l=lp=2; l=2,lp=0; l=0,lp=2; and l=lp=0 terms)
#################################################################
# First define tensors (see notes)
def T(E,pvec,kvec):   # Cartesian basis
  k=norm(kvec); p=norm(pvec)
  beta_k = k/(E-omega(k)); gam_k = sqrt(1/(1-beta_k**2))
  beta_p = p/(E-omega(p)); gam_p = sqrt(1/(1-beta_p**2))
  khat = [ki/k for ki in kvec] if k!=0 else kvec
  phat = [pi/p for pi in pvec] if p!=0 else pvec

  t = np.zeros((3,3))
  for i in range(0,3):
    t[i][i] = -1
    for j in range(0,3):
      t[i][j] += khat[i]*phat[j] * ( gam_k*beta_k*gam_p*beta_p - mydot(khat,phat)*(gam_k-1)*(gam_p-1) ) - khat[i]*khat[j]*(gam_k-1) - phat[i]*phat[j]*(gam_p-1)
  return t

# Convert Cartesian tensor to rank-2 spherical tensor (see notes)
# (TB: sqrt(1/30) should probably be pulled out of the definition)
def sph2complex(t,m): # opposite conjugation convention as Steve
  if m==2:
    return sqrt(1/30)*( t[0][0]-t[1][1] + 1j*(t[0][1]+t[1][0]) )
  elif m==-2:
    return sqrt(1/30)*( t[0][0]-t[1][1] - 1j*(t[0][1]+t[1][0]) )
  elif m==1:
    return sqrt(1/30)*(-t[0][2]-t[2][0] - 1j*(t[1][2]+t[2][1]) )
  elif m==-1:
    return sqrt(1/30)*( t[0][2]+t[2][0] - 1j*(t[1][2]+t[2][1]) )
  elif m==0:
    return sqrt(1/45)*( 2*t[2][2]-t[0][0]-t[1][1] )
  else:
    print('Error: invalid m in sphT2')

def sph2real(t,m): # opposite conjugation convention as Steve
  if m==2:
    return sqrt(1/15)*( t[0][0]-t[1][1] )
  elif m==-2:
    return sqrt(1/15)*( t[0][1]+t[1][0] )
  elif m==1:
    return sqrt(1/15)*( t[0][2]+t[2][0] )
  elif m==-1:
    return sqrt(1/15)*( t[1][2]+t[2][1] )
  elif m==0:
    return sqrt(1/45)*( 2*t[2][2]-t[0][0]-t[1][1] )
  else:
    print('Error: invalid m in sphT2')

def sph2(t,m,Ytype):
  if Ytype=='real' or Ytype=='r':
    return sph2real(t,m)
  else:
    return sph2complex(t,m)

# Coefficient defined by sph2(t,m) = sphcoeff(m,i,j)*t[i][j]
def sphcoeff_complex(m,i,j): # opposite conjugation convention as Steve
  if m==2 or m==-2:
    if i==j==0 or i==j==1:
      return (-1)**i*sqrt(1/30)
    elif i==j==1:
      return -sqrt(1/30)
    elif (i==0 and j==1) or (i==1 and j==0):
      return np.sign(m)*1j*sqrt(1/30)
    else:
      return 0

  elif m==1 or m==-1:
    if (i==0 and j==2) or (i==2 and j==0):
      return -np.sign(m)*sqrt(1/30)
    elif (i==1 and j==2) or (i==2 and j==1):
      return -1j*sqrt(1/30)
    else:
      return 0

  elif m==0:
    if i==j==2:
      return 2*sqrt(1/45)
    elif i==j==0 or i==j==1:
      return -sqrt(1/45)
    else:
      return 0

  else:
    print('Error: invalid m in sph2coeff')

def sphcoeff_real(m,i,j):
  if m==2 and (i==j==0 or i==j==1):
      return (-1)**i*sqrt(1/15)
  elif m==-2 and ( (i==0 and j==1) or (i==1 and j==0) ):
      return sqrt(1/15)
  elif m==1 and ( (i==0 and j==2) or (i==2 and j==0) ):
      return sqrt(1/15)
  elif m==-1 and ( (i==1 and j==2) or (i==2 and j==1) ):
      return sqrt(1/15)
  elif m==0:
    if i==j==2:
      return 2*sqrt(1/45)
    elif i==j==0 or i==j==1:
      return -sqrt(1/45)
    else:
      return 0
  elif -2<=m<=2:
    return 0
  else:
    print('Error: invalid m in sph2coeff')

def sphcoeff(m,i,j,Ytype='r'):
  if Ytype=='real' or Ytype=='r':
    return sphcoeff_real(m,i,j)
  else:
    return sphcoeff_complex(m,i,j)

# Now the matrix element
def wddmm(E,pvec,lp,mp,kvec,l,m,Ytype='r'):
  k=norm(kvec); p=norm(pvec)
  t=T(E,pvec,kvec)

  if l==m==lp==mp==0:
    S00 = sum([t[i][j]**2 for i in range(0,3) for j in range(0,3)])
    return 4/9*(qst(E,k)*qst(E,p))**2 * S00

  elif l==2 and lp==mp==0:
    S20 = np.dot(t,np.transpose(t))
    #return 4/3*(qst(E,k)*qst(E,p))**2 * conj(sph2(S20,m,Ytype))
    return 4/3 * qst(E,p)**2 * conj(sph2(S20,m,Ytype)) # TB, no q (of k)

  elif lp==2 and l==m==0:
    S02 = np.dot(np.transpose(t),t)
    #return 4/3*(qst(E,k)*qst(E,p))**2 * sph2(S02,mp,Ytype)
    return 4/3*qst(E,k)**2 * sph2(S02,mp,Ytype) # TB, no q (of p)

  elif l==lp==2:
    # TB: This method is probably a tad slower than listing all cases, but not by much (speed seems fine to me)
    s=0
    for i1 in range(0,3):
      for i2 in range(0,3):
        for j1 in range(0,3):
          for j2 in range(0,3):
            s += conj(sphcoeff(m,i1,i2,Ytype))*sphcoeff(mp,j1,j2,Ytype) * t[i1][j1]*t[i2][j2]
    #return 4*(qst(E,k)*qst(E,p))**2 * s
    return 4*s # TB, no q
  else:
    return 0

#################################################################
# Putting it all together
#################################################################
# First get wdd and p2sum
#@jit(fastmath=True,cache=True)
@jit(fastmath=True,cache=True)
def wdd(E,pvec,lp,mp,kvec,l,m,Ytype='r'):
  if l==m==lp==mp==0:
    return wddpp(E,pvec,lp,mp,kvec,l,m) + wddmp(E,pvec,lp,mp,kvec,l,m,Ytype) + wddpm(E,pvec,lp,mp,kvec,l,m,Ytype) + wddmm(E,pvec,lp,mp,kvec,l,m,Ytype)
  elif l==2 and lp==mp==0:
    return wddmp(E,pvec,lp,mp,kvec,l,m,Ytype) + wddmm(E,pvec,lp,mp,kvec,l,m,Ytype)
  elif l==m==0 and lp==2:
    return wddpm(E,pvec,lp,mp,kvec,l,m,Ytype) + wddmm(E,pvec,lp,mp,kvec,l,m,Ytype)
  elif l==lp==2:
    return wddmm(E,pvec,lp,mp,kvec,l,m,Ytype)
  else:
    return 0

@jit(fastmath=True,cache=True)
def p2sum(E,pvec,lp,mp,kvec,l,m,Ytype='r'):
  if l==m==lp==mp==0:
    return wss(E,pvec,lp,mp,kvec,l,m) + wds(E,pvec,lp,mp,kvec,l,m,Ytype) + wsd(E,pvec,lp,mp,kvec,l,m,Ytype) + wdd(E,pvec,lp,mp,kvec,l,m,Ytype)
  elif l==2 and lp==mp==0:
    return wds(E,pvec,lp,mp,kvec,l,m,Ytype) + wdd(E,pvec,lp,mp,kvec,l,m,Ytype)
  elif l==m==0 and lp==2:
    return wsd(E,pvec,lp,mp,kvec,l,m,Ytype) + wdd(E,pvec,lp,mp,kvec,l,m,Ytype)
  elif l==lp==2:
    return wddmm(E,pvec,lp,mp,kvec,l,m,Ytype)
  else:
    return 0

# Finally find K3B
@jit(fastmath=True,cache=True)
def K3B(E,pvec,lp,mp,kvec,l,m,Ytype='r'):
  if l==m==lp==mp==0:
    const = -4*(2*E**2-9)
  else:
    const=0
  return 4*p2sum(E,pvec,lp,mp,kvec,l,m,Ytype) + const
