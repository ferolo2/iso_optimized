from defns import omega, E2k, qst, y2
import numpy as np, sys
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg; norm=LA.norm

# Relevant boosted momenta
def pstark(E,pvec,kvec): # (see notes)
  k=LA.norm(kvec); p=LA.norm(pvec)
  beta_k = k/(E-omega(k)); gam_k = sqrt(1/(1-beta_k**2))
  khat = [ki/k for ki in kvec] if k!=0 else kvec

  return [pvec[i] + ( gam_k*beta_k*omega(p) + (gam_k-1)*np.dot(pvec,khat) ) * khat[i] for i in range(0,3)]

def p12stark(E,pvec,kvec): # (see notes)
  k=LA.norm(kvec); p=LA.norm(pvec)
  beta_k = k/(E-omega(k)); gam_k = sqrt(1/(1-beta_k**2))
  khat = [ki/k for ki in kvec] if k!=0 else kvec

  return [-pvec[i] + ( gam_k*beta_k*(E-omega(p)) - (gam_k-1)*np.dot(pvec,khat) ) * khat[i] for i in range(0,3)]

# Relevant tensors
def T(E,pvec,kvec):   # Cartesian basis
  k=LA.norm(kvec); p=LA.norm(pvec)
  beta_k = k/(E-omega(k)); gam_k = sqrt(1/(1-beta_k**2))
  beta_p = p/(E-omega(p)); gam_p = sqrt(1/(1-beta_p**2))
  khat = [ki/k for ki in kvec] if k!=0 else kvec
  phat = [pi/p for pi in pvec] if p!=0 else pvec

  t = np.zeros((3,3))
  for i in range(3):
    t[i,i] = -1
    for j in range(3):
      t[i,j] += khat[i]*phat[j] * ( gam_k*beta_k*gam_p*beta_p - np.dot(khat,phat)*(gam_k-1)*(gam_p-1) ) - khat[i]*khat[j]*(gam_k-1) - phat[i]*phat[j]*(gam_p-1)
  return t

# Convert Cartesian tensor to rank-2 spherical tensor (see notes)
def sph2real(t,m): # TB: note that I leave in the sqrt(1/15) factor (Steve does not)
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

def sph2(t,m,Ytype='r'):
  if Ytype=='real' or Ytype=='r':
    return sph2real(t,m)
  else:
    return sph2complex(t,m)

# Coefficient defined by sph2(t,m) = sphcoeff(m,i,j)*t[i,j]
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

# Spherical tensor element of S_{ik,jn} = T_{ij}T_{jn}
def Sspher(t,lp,mp,l,m):
  if lp==l==mp==m==0:
    return sum([t[i,j]**2 for i in range(3) for j in range(3)])
  elif lp==mp==0 and l==2:
    return sph2real(np.dot(t,np.transpose(t)),m)
  elif lp==2 and l==m==0:
    return sph2real(np.dot(np.transpose(t),t),mp)
  elif lp==l==2:
    out = 0
    for i1 in range(3):
      for i2 in range(3):
        for j1 in range(3):
          for j2 in range(3):
            out += sphcoeff_real(m,i1,i2)*sphcoeff_real(mp,j1,j2) * t[i1,j1]*t[i2,j2]
    return out

# Spherical tensor element of U_{ik,jn} = T_{ij}u_{kn}, where u_{kn} = (p12t*k)_k (p12*p)_n
def Uspher(t,u,lp,mp,l,m):
  if lp==l==mp==m==0:
    return sum([t[i,j]*u[i,j] for i in range(3) for j in range(3)])
  elif lp==mp==0 and l==2:
    return sph2real(np.dot(t,np.transpose(u)),m)
  elif lp==2 and l==m==0:
    return sph2real(np.dot(np.transpose(t),u),mp)
  elif lp==l==2:
    out = 0
    for i1 in range(3):
      for i2 in range(3):
        for j1 in range(3):
          for j2 in range(3):
            out += sphcoeff_real(m,i1,i2)*sphcoeff_real(mp,j1,j2) * t[i1,j1]*u[i2,j2]
    return out



###############################################################
# Complex basis versions
###############################################################
# (TB: sqrt(1/30) could be pulled out of the definition)
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
