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
import itertools
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from K3df import K3cubicB as K33B
from K3df import K3quad as K3
from defns import list_nnk,chop
from scipy import interpolate


# Full K33B mat
@jit(fastmath=True,cache=True)
def F3i_00(e,L,a0,IPV):

  K3 = 0.
  f3mat= F3_mat.F3mat00(e,L,a0,0.,0.,0.,0.5,IPV)
#  resaux = sorted(np.linalg.eig(pr.iso_proj00(f3mat,e,L))[0],key=abs)
#  print(1./(np.ones(len(f3mat))@f3mat@np.ones(len(f3mat))) + K3)
  return 1./(np.ones(len(f3mat))@f3mat@np.ones(len(f3mat))) + K3



#@jit(fastmath=True,cache=True)
def EWrange0(L,a0,IPV,energies):
    res=np.zeros(len(energies))
    lenergies= len(list(energies))
    aux = []
    for i in range(lenergies):
        aux.append((energies[i],L,a0,IPV))

    result=[]
    print(energies)
#    result = list(itertools.starmap(F3i_00, aux))
    result = pool.starmap(F3i_00, aux)
#    for i in prange(lenergies):
#      print(F3i_00(energies[i],L,a0,IPV))
#      result.append(F3i_00(energies[i],L,a0,IPV))

    return result

  
def Ethres(L,a):
    I = -8.914
    J = I**2+16.532
    E3 = 12*math.pi*a/L**3*(1 - (a/math.pi/L)*I +(a/math.pi/L)**2*J )
    return E3



#@jit(fastmath=True,cache=True)
def findE(Etest,dEtest,L):
  
  energies= [Etest-dEtest, Etest, Etest+dEtest]
  for i in range(5):
    print(energies)
    res = EWrange0(L,a,IPV,energies)
    print(res)
    resaux=res
    resaux=np.array(res)/max(res)

    if(res[0]>res[1]):
        resaux=np.dot(-1.,np.array(res))/max(res)


        
    Ener=np.interp(0,resaux,energies)
    
    print('E=%.12f' % Ener )
   
    
    if(res[0]/res[1]<0):
        energies[2]=energies[1]
        energies[1]=Ener

    elif(res[1]/res[2]<0):
        energies[0]=energies[1]
        energies[1]=Ener
    else:
        energies[0] = energies[0] - abs(energies[0]-energies[1])*1.5
        energies[2] = energies[2] + abs(energies[2]-energies[1])*1.5
        print('error')
#        exit()

    dEner = min([abs(energies[0]-Ener),abs(energies[2]-Ener)])

    energies[0] = Ener-dEner
    energies[2] = Ener+dEner
  return Ener






#listL=np.array(range(150,155))*0.1
listL=np.array(range(150,100,-5))*0.1
#listL=[15,15.1]
a = 6.0
#IPV=0
IPV=-1.


Etest = 1+2*np.sqrt(1.-1./a**2)
MB = 2*np.sqrt(1.-1./a**2)

#Etest = 2.87876
#Etest = 2.9839
#Etest=2.73226
#Etest = 2.83365
#Etest = 2.959609
#Etest = 2.97190
#Etest = 2.6931
#Etest=2.96405
#Etest = 2.9611824 #2.9611833552 #2.961184
dEtest = 0.005
start = time.time()


#print(F3i_00(Etest,19.1,a,IPV))

#import matplotlib.pyplot as plt
sols = []

#aux = []
#for i in range(len(listL)):
#  aux.append((Etest,dEtest,listL[i]))
#sols = list(itertools.starmap(findE,aux))

print(listL)
#dEaux = 2.979888710842 - np.sqrt(1+2*(2*math.pi/59.)**2) - np.sqrt(MB**2 + 2*(2*math.pi/59.)**2)
dEaux = 0.005
#dEaux = 0.00055
pool = Pool(processes=3)
Etest = 2.986
#dEtest= 0.00003
for L in listL:
#  Etest = np.sqrt(1+1*(2*math.pi/L)**2) + np.sqrt(MB**2 + 1*(2*math.pi/L)**2) + dEaux
  print(Etest)
  Ener=  findE(Etest,dEtest,L)
  sols.append(Ener)
  Etest =Ener
#  dEaux = Ener - np.sqrt(1+1*(2*math.pi/L)**2) - np.sqrt(MB**2 + 1*(2*math.pi/L)**2)
pool.close
#print(listL)
#print(sols)

print('#L E')
for i in range(len(listL)):
  print(listL[i],sols[i])

end = time.time()
print('time is:', end - start, ' s')
