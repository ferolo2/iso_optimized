from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
    import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
sys.path.insert(0,cwd+'/K3df')
from F3 import H_mat,F3_mat,Gmatrix
import math
import numpy as np
import time
import projections as pr
from numba import jit,njit,prange,autojit
from multiprocessing import Pool
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from K3df import K3cubicB as K33B
from K3df import K3quad as K3
from defns import list_nnk,chop
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib
#from matplotlib import colors


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def F3i_00(e,L,a0,IPV):
    
#  f3mat= F3_mat.F3imat00(e,L,a0,0.,0.,0.,0.3,IPV)
#    r0=8.34
    r0=0.
    f3mat= F3_mat.F3mat00(e,L,a0,r0,0.,0.,0.5,IPV)
    resaux = sorted(np.linalg.eig(pr.iso_proj00(f3mat,e,L))[0],key=abs)
#  print(len(pr.iso_proj00(f3mat,e,L)))
#  return np.ndarray.flatten(pr.iso_proj00(f3mat,e,L))[0]
#  return np.linalg.det(f3mat)
#  return np.prod(resaux[0:2])
#  print(np.ones(len(pr.iso_proj00(f3mat,e,L)))@pr.iso_proj00(f3mat,e,L)@np.ones(len(pr.iso_proj00(f3mat,e,L))))
#  return 1./(np.ones(len(pr.iso_proj00(f3mat,e,L)))@pr.iso_proj00(f3mat,e,L)@np.ones(len(pr.iso_proj00(f3mat,e,L))))
    return 1./(np.ones(len(f3mat))@f3mat@np.ones(len(f3mat)))

result=[1]

L=4.0
IPV=-1
a0=2
#IPV=0.
Etest =1.3
#Etest=3.40
dEtest = 0.00101/40
nmax = 1700*40
Eaux = 3.00014 # + 12*math.pi*a0/(L**3)*1.0034
Erange = np.linspace(Etest, Etest + (nmax-1)*dEtest, num=nmax, endpoint=True)
pool = Pool(processes=5)
lenergies= len(list(Erange))
aux = []
for i in range(lenergies):
  aux.append((Erange[i],L,a0,IPV))

result = pool.starmap(F3i_00, aux)
#print(Erange)
print(1/np.array(result))

  
#matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"].append(r'\usepackage{xfrac}')

fig, ax = plt.subplots()
ax.plot(Erange,result,'.',label='$a_0=%5.2f$, $I_{PV}=-1, L=%d$' % (a0,L), linewidth=2,color='green')

ax.set_ylabel('$\\frac{1}{F^{iso}_3}$  ', fontsize=24,rotation=0,labelpad=29)
ax.set_xlabel('$E/m$', fontsize=21)
plt.gcf().subplots_adjust(left=0.2)
plt.gcf().subplots_adjust(bottom=0.15)

#plt.xlim([-2,0])
plt.ylim([-300000,300000])


#a0=-1*a0
#a0=100
aux = []
for i in range(lenergies):
  aux.append((Erange[i],L,a0,IPV))
start = time.time() 
#result = pool.starmap(F3i_00, aux)
end = time.time()
print('time is:', end - start, ' s')

#ax.plot(Erange,result,'.',label='$a_0=%5.2f$, $I_{PV}=-1, L=%d$' % (a0,L), linewidth=2,color='red')

#L=40

zeroline = np.linspace(Etest, Etest + (nmax-1)*dEtest, num=1000, endpoint=True)

ax.plot(zeroline,0*zeroline,'-',color='black')
#ax.plot(zeroline,-80000 + 0*zeroline,'-',color='black')
#ax.plot(zeroline,-120000 + 0*zeroline,'-',color='black')
#ax.plot(zeroline,40000 + 0*zeroline,'-',color='black')
#ax.plot(zeroline,60000 + 0*zeroline,'-',color='black')

#K1 = result[find_nearest( Erange, Eaux )]
#ax.plot(zeroline,0*zeroline + K1,'--',color='red')
#a0 = 1000
L=25
aux = []
for i in range(lenergies):
  aux.append((Erange[i],L,a0,IPV))


start = time.time() 
#result = pool.starmap(F3i_00, aux)
#end = time.time()
print('time is:', end - start, ' s')

#print(1/np.array(result))
#ax.plot(Erange,result,'.',label='$a_0=%5.2f$, $I_{PV}=-1, L=%d$' % (a0,L), linewidth=2,color='blue')



#a0 = 10000
L=30
aux = []
for i in range(lenergies):
  aux.append((Erange[i],L,a0,IPV))


start = time.time() 
#result = pool.starmap(F3i_00, aux)
end = time.time()
print('time is:', end - start, ' s')

#print(1/np.array(result))
#ax.plot(Erange,result,'.',label='$a_0=%5.2f$, $I_{PV}=-1, L=%d$' % (a0,L), linewidth=2,color='magenta')



K1p = result[find_nearest( Erange, Eaux )]
Eaux2 = Erange[find_nearest( Erange, Eaux )]
#ax.plot(zeroline,0*zeroline + K1p,'--',color='blue')

#print('maching at L=%f' % L,Eaux2, K1,K1p)


xcoords = [3., 1+2*np.sqrt(1+(2*math.pi/L)**2)]
xcoords = [3.]
xcoords = [3., 1+2*np.sqrt(1+(2*math.pi/L)**2), 1+2*np.sqrt(1+2*(2*math.pi/L)**2)]
#for xc in xcoords:
#    plt.axvline(x=xc,linestyle='--',color='gray')

#plt.axvline(x=1+2*np.sqrt(1-1/a0**2),linestyle='--',color='magenta',label='1+$m_B$')
#
    
ax.legend(loc='lower left', fontsize=10, ncol=1, numpoints=1)



plt.tight_layout()
plt.savefig('FvsE_varya0.pdf',bbox_inches='tight', pad_inches = 0) 
plt.show()





exit()

  
