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


# Full K33B mat




#K3mat(E,L,K0,K1,K2,A,B,Ytype='r')
    

def F3i(e,L,a0,a2):

  K0 = -1000*0
  K1 = -100*0
  K2 = -50*0
  A = -470*0
  B = 0.
  Kdf = K3.K3mat(e,L,K0,K1,K2,A,B)
  f3mat= F3_mat.F3i_mat(e,L,a0,0.,0.,a2,0.3)
  return pr.irrep_proj(f3mat + Kdf ,e,L,"E+")


def K3pF3i(e,L,a0,a2):
    return F3i(e,L,a0,a2)+100000*K33Bmat(e,L,a0,a2)


  
def H2(e,L,a,a2):
    return H_mat.Hmat(e,L,a,0.,0.,a2,0.5)
def H2proj(e,L,a,a2):
    return pr.irrep_proj(H_mat.Hmat(e,L,a,0.,0.,a2,0.5),e,L,"A1+")
def H0(e,L,a,a2):
    return H_mat.Hmat00(e,L,a,0.,0.,0.5)

def EWrange2(L,a,a2,energies):
    res=np.zeros(len(energies))
    lenergies= len(list(energies))
    pool = Pool(processes=3)

    aux = []
    for i in range(lenergies):
        aux.append((energies[i],L,a,a2))

    res=[]    
#    print(aux)
#    exit()
    
#    result = pool.starmap(F3i, [(energies[0],L,a,a2),(energies[1],L,a,a2),(energies[2],L,a,a2)])
#    result = pool.starmap(K33Bmat, aux)
    result = pool.starmap(F3i, aux)
#    result = map(K33Bmat, aux)
#    result = pool.starmap(H0, [(energies[0],L,a,a2),(energies[1],L,a,a2),(energies[2],L,a,a2)])
    for i in prange(lenergies):
#        print(np.linalg.eig(result[i])[0])
#        res[i] = sorted(np.linalg.eig(result[i])[0])[-1].real  
        resaux = sorted(np.linalg.eig(result[i])[0])
#        res.append([energies[i], np.prod([resaux[0],resaux[2],resaux[4],resaux[6],resaux[8],resaux[10],resaux[12]]).real])   
        res.append( np.prod(resaux).real)
        print(resaux)
#        print(resaux,len(resaux))
#        res.append([energies[i],resaux[0].real,resaux[1].real,resaux[2].real,resaux[3].real,resaux[4].real,resaux[5].real])
#        res.append([energies[i],resaux[0].real,resaux[1].real,resaux[2].real,resaux[3].real,resaux[4].real,resaux[5].real,resaux[6].real,resaux[7].real,resaux[8].real,resaux[9].real])
#        res.append([energies[i], resaux[0].real])
#        print(resaux)
#        print(resaux[0:10])

#        print(resaux)
#        res[i] = np.prod(resaux[-4:1]).real
#        res.append( np.prod(resaux[0:5]).real)
#        res[i] = np.prod(resaux[0:2]).real

        #print("EV ",resaux[0:2])
    return res



def EWrange0(L,a,a2,energies):
    res=np.zeros(len(energies))
    lenergies= len(list(energies))
    pool = Pool(processes=4)

    aux = []
    for i in range(lenergies):
        aux.append((energies[i],L,a,a2))

    res=[]    
#    print(aux)
#    exit()
    
#    result = pool.starmap(F3i, [(energies[0],L,a,a2),(energies[1],L,a,a2),(energies[2],L,a,a2)])
#    result = pool.starmap(K33Bmat, aux)
    result = pool.starmap(F3i, aux)
#    result = map(K33Bmat, aux)
#    result = pool.starmap(H0, [(energies[0],L,a,a2),(energies[1],L,a,a2),(energies[2],L,a,a2)])
    for i in prange(lenergies):
#        print(np.linalg.eig(result[i])[0])
#        res[i] = sorted(np.linalg.eig(result[i])[0])[-1].real  
        resaux = sorted(np.linalg.eig(result[i])[0])
#        res.append([energies[i],resaux[0].real,resaux[1].real,resaux[2].real,resaux[3].real,resaux[4].real,resaux[5].real])
#        res.append([energies[i],resaux[0].real,resaux[1].real,resaux[2].real,resaux[3].real,resaux[4].real,resaux[5].real,resaux[6].real,resaux[7].real,resaux[8].real,resaux[9].real])
        res.append([energies[i], np.prod(resaux).real])

#        print(resaux)
#        res[i] = np.prod(resaux[-4:1]).real
#        res.append( np.prod(resaux[0:5]).real)
#        res[i] = np.prod(resaux[0:2]).real

        #print("EV ",resaux[0:2])
    return res



  
def Ethres(L,a):
    I = -8.914
    J = I**2+16.532
    E3 = 12*math.pi*a/L**3*(1 - (a/math.pi/L)*I +(a/math.pi/L)**2*J )
    return E3






  



L=9
a = 0.2
a2 = 1.0
Etest = 2*np.sqrt(1+(2*math.pi/L)**2)+1+0.05
#Etest = 3.237
dEtest = 0.008

print('Efree = ', 2*np.sqrt(1+(2*math.pi/L)**2)+1)
start = time.time()

energies =[]

Estart = 3.58
dE = 0.006



for i in range(32):
  energies.append(Estart+i*dE)


pool = Pool(processes=4)
aux = []
for i in range(len(energies)):
  aux.append((energies[i],L,a,a2))


print(energies)

result = pool.starmap(F3i, aux)


res=[]
for i in range(len(result)):    
  resaux = sorted(np.linalg.eig(result[i])[0])
  #res.append( np.prod(resaux).real)
  res.append( np.prod([resaux[0],resaux[2],resaux[4],resaux[6],resaux[8],resaux[10],resaux[12]]).real)




print(res)

        
end = time.time()
print('time is:', end - start, ' s')



import matplotlib.pyplot as plt


plt.plot(energies,res/np.absolute(res),'o')

plt.show()


aux = res/np.absolute(res)
for i in range(len(res)-1):

  if  aux[i]/aux[i+1]==-1:
    print('possible state between ', energies[i],' and ',energies[i+1])
    energiesaux=[energies[i],energies[i] + (energies[i+1]-energies[i])/2,energies[i+1]]
    EWrange2(L,a,a2,energiesaux)

  
