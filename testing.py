import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
from F3 import H_mat
from F3 import sums_alt as sums
from F3 import F2_alt as F2
from F3 import K2i_mat as K2
from F3 import Gmatrix as Gm
import defns
import math
import numpy as np
import time
from numba import jit,njit,prange,autojit,vectorize
from F3 import sums_alt as sums
from scipy.special import erfc,erfi,erf
from scipy.optimize import fsolve
from joblib import Parallel, delayed
from numpy.lib.scimath import sqrt

npsqrt = np.sqrt
exp = np.exp
def getnmax(cutoff,alpha,x2,gamma):
    eqn = lambda l : -cutoff + 2*math.pi*npsqrt(math.pi/alpha) * exp(alpha*x2)*erfc(npsqrt(alpha)*l)
    n0=1
    solution=fsolve(eqn, n0)
    print(max(solution*gamma,1)+3)

    return int(np.round(max(solution*gamma,1)+3))



L=84
E=2.9972794820231976
print(len(defns.list_nnk(E,L)))

exit()



a0=25
r0=8.34
E=3.1
L=10
k=2.25
IPV=-1
print(K2.K2inv(E,np.array([0.,0.,k*2*math.pi/L]),0,0,a0,r0,0.,0.,IPV))
    

exit()


L=4.5
E=np.sqrt(1+(2*math.pi/L)**2)*2+1

print(defns.shell_list(E,L))
print(defns.kmax(E)/(2*math.pi/L))

E=3.262


print(defns.shell_list(E,L))
print(defns.kmax(E)/(2*math.pi/L))


print(sums.hh(E,np.sqrt(2**2+2**2)*2*math.pi/L ))

print(defns.kmax(E)/(2*math.pi/L))
print(np.sqrt(5.))
exit()
start = time.time()


aux = 0
for i in range(-30,30+1):
    for j in range(-30,30+1):
       aux += i**2+j**2

print(aux)

end = time.time()
print('time is:', end - start, ' s')
start = time.time()


aux=0
for i in range(-30,30+1):
    for j in range(-30,i+1):
        if(i==j):
            aux += i**2+j**2
        else:
            aux += 2*(i**2+j**2)
print(aux)


end = time.time()

print('time is:', end - start, ' s')

L=20
E=3.01
nnk = np.array([2,2,1.])
nk = sums.norm(nnk)
alpha=0.5
gamma = sums.gam(E,npsqrt(1+4+9)*2*math.pi/L)
x2= sums.xx2(E, L, npsqrt(1+4+9)*2*math.pi/L)
x2= sums.xx2(E, L, 0*npsqrt(1+4+9)*2*math.pi/L)
cutoff=1e-9
start = time.time()
#Parallel(n_jobs=4)(delayed(sums.sum_nnk)(E, L,nnk,2,0,2,0,alpha) for i in range(1000))

print(sums.sum_nnk(E, L,nnk,2,2,2,2,alpha))

end = time.time()
print('time is:', end - start, ' s')
start = time.time()


print(sums.sum_aab(E, L,nnk,2,2,2,2,alpha))
end = time.time()
print('time is:', end - start, ' s')

exit()
print(sums.summand(E, L, np.array([1.,0,0]), nnk, nk, gamma, x2,0,0,0,0,alpha))
print(sums.summand(E, L, np.array([0,1.,0]), nnk, nk, gamma, x2,0,0,0,0,alpha))
print(sums.summand(E, L, np.array([0.,0,1.]), nnk, nk, gamma, x2,0,0,0,0,alpha))
print(sums.summand(E, L, np.array([-1.,0,0]), nnk, nk, gamma, x2,0,0,0,0,alpha))
print(sums.summand(E, L, np.array([0,-1.,0]), nnk, nk, gamma, x2,0,0,0,0,alpha))
print(sums.summand(E, L, np.array([0.,0,-1.]), nnk, nk, gamma, x2,0,0,0,0,alpha))
print(sums.summand(E, L, np.array([0.,1.,1.]), nnk, nk, gamma, x2,0,0,0,0,alpha))
print(sums.summand(E, L, np.array([1.,1.,0.]), nnk, nk, gamma, x2,0,0,0,0,alpha))
print(sums.summand(E, L, np.array([1.,0.,1.]), nnk, nk, gamma, x2,0,0,0,0,alpha))



print('hola')
nnk = np.array([1,1,0.])
nk = sums.norm(nnk)

print(sums.summand(E, L, np.array([1.,0,0]), nnk, nk, gamma, x2,2,2,2,-2,alpha))
print(sums.summand(E, L, np.array([0,1.,0]), nnk, nk, gamma, x2,2,2,2,-2,alpha))

print(sums.sum_nnk(E, L, nnk,2,2,2,-2,alpha))

exit()

print(sums.summand(E, L, np.array([1.,2.,0]), nnk, nk, gamma, x2,2,0,2,0,alpha))
print(sums.summand(E, L, np.array([2.,1.,0]), nnk, nk, gamma, x2,2,0,2,0,alpha))
print(sums.summand(E, L, np.array([0.,0,1.]), nnk, nk, gamma, x2,2,0,2,0,alpha))
print(sums.summand(E, L, np.array([0.,0,-1.]), nnk, nk, gamma, x2,2,0,2,0,alpha))
print(sums.summand(E, L, np.array([0.,1.,1.]), nnk, nk, gamma, x2,2,0,2,0,alpha))
print(sums.summand(E, L, np.array([1.,1.,0.]), nnk, nk, gamma, x2,2,0,2,0,alpha))
print(sums.summand(E, L, np.array([1.,0.,1.]), nnk, nk, gamma, x2,2,0,2,0,alpha))

end = time.time()
print('time is:', end - start, ' s')
exit()

start = time.time()
for i in range(1000):
    sums.sum_nnk(E, L,nnk,2,0,2,0,alpha)
end = time.time()
print('time is:', end - start, ' s')
