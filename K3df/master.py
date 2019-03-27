#from K3df 
import K3A, K3B, K3quad, defns
y2=defns.y2
import numpy as np, time
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg
# 1j = imaginary number

####################################################################################
# Playground file to test functions
# Note: input order for all matrix elements is (E,outgoing momenta,incoming momenta)
####################################################################################

####################################################################################
# Sample matrix element calculations
####################################################################################
# Input parameters for K3A, K3B, and K3quad
E=3.05

L=80 # lattice not necessary for K3quad (only need kvec and pvec)
nkvec=[1,2,-1]; npvec=[2,0,1] 
kvec=[i*2*pi/L for i in nkvec]; pvec = [i*2*pi/L for i in npvec]
k=LA.norm(kvec); p=LA.norm(pvec)

l=2; m=1; lp=0; mp=0
K0=0; K1=0; K2=0; A=1; B=1
Ytype='r' # spherical harmonic type: 'r' for real, 'c' for complex

####################################################################################
# Outputs w/ runtimes
t1=time.time()
print('Delta    = %f' % (E**2-9))
print('Delta**2 = %f' % (E**2-9)**2)
print('K3A      = {:f}'.format(K3A.K3A(E,pvec,lp,mp,kvec,l,m,Ytype)))
print('K3B      = {:f}'.format(K3B.K3B(E,pvec,lp,mp,kvec,l,m,Ytype)))
print('runtime: %fs \n' % (time.time()-t1))

t1=time.time()
print('K3quad   = {:f}'.format(K3quad.K3quad(E,pvec,lp,mp,kvec,l,m,K0,K1,K2,A,B,Ytype)))
print('runtime: %fs \n' % (time.time()-t1))


####################################################################################
# Check that complex and real spherical harmonic bases are consistent
####################################################################################
t=[[1,-2,3],[4,5,6],[-7,8,9]]

# Check consistency of spherical tensor decomposition
s1=0
for i in range(0,3):
  s1 -= 1/3*LA.norm(kvec)**2 * t[i][i]
  for j in range(0,3):
    s1 += kvec[i]*kvec[j] * t[i][j]
s2=0
for m in range(-2,3):
  s2 += y2(kvec,m,'c') * conj(K3B.sph2(t,m,'c'))
s3=0
for m in range(-2,3):
  s3 += y2(kvec,m,'r') * K3B.sph2(t,m,'r')

print('Spherical tensor check')
print('direct:  %f' % s1)
print('complex: {:f}'.format(s2))
print('real:    %f' % s3); print()

# Check consistency of full function K3B (not just matrix element)
S1=K3B.K3B(E,pvec,0,0,kvec,0,0,'c')
for m in range(-2,3):
  S1 += y2(kvec,m,'c')/k**2 * K3B.K3B(E,pvec,0,0,kvec,2,m,'c')
  for mp in range(-2,3):
    S1 += y2(kvec,m,'c')*conj(y2(pvec,mp,'c'))/(k*p)**2 * K3B.K3B(E,pvec,2,mp,kvec,2,m,'c')
for mp in range(-2,3):
  S1 += conj(y2(pvec,mp,'c'))/p**2 * K3B.K3B(E,pvec,2,mp,kvec,0,0,'c')

S2=K3B.K3B(E,pvec,0,0,kvec,0,0,'r')
for m in range(-2,3):
  S2 += y2(kvec,m,'r')/k**2 * K3B.K3B(E,pvec,0,0,kvec,2,m,'r')
  for mp in range(-2,3):
    S2 += y2(kvec,m,'r')*y2(pvec,mp,'r')/(k*p)**2 * K3B.K3B(E,pvec,2,mp,kvec,2,m,'r')
for mp in range(-2,3):
  S2 += y2(pvec,mp,'r')/p**2 * K3B.K3B(E,pvec,2,mp,kvec,0,0,'r')

print('K3B check (full function)')
print('complex: {:f}'.format(S1))
print('real:    %f' % S2)