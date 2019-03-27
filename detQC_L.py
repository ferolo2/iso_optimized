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
from defns import list_nnk,chop



# Full K33B mat

def K33Bmat(E,L,a0,a2):
  nnk_list = list_nnk(E,L)
  N = len(nnk_list)
  #print(nnk_list)                                                                                                                                                                       
  #print(N)                                                                                                                                                                              
  Gfull = []
  for p in range(N):
    nnp = list(nnk_list[p])
    Gp = []
    for k in range(N):
      nnk = list(nnk_list[k])

      Gpk = np.zeros((6,6))
      for i1 in range(6):
        [l1,m1] = Gmatrix.lm_idx(i1)
        for i2 in range(6):
          [l2,m2] = Gmatrix.lm_idx(i2)
          Gpk[i1,i2] = K33B.K3cubicB(E,2*math.pi/L*np.array(nnp),l1,m1,2*math.pi/L*np.array(nnk),l2,m2)

      Gp.append(Gpk)

      
    Gfull.append(Gp)


  matrix =  np.ones_like(chop(np.block(Gfull)))

  print(matrix.shape)

  
  return pr.irrep_proj(matrix,E,L,"A1")
#  return pr.irrep_proj(chop(np.block(Gfull)),E,L,"A1")

    

def F3i(e,L,a0,a2):
    f3mat= F3_mat.F3i_mat(e,L,a0,0.,0.,a2,0.3)
    return pr.irrep_proj(f3mat,e,L,"A2+")


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
    result = pool.starmap(H2proj, aux)
#    result = map(K33Bmat, aux)
#    result = pool.starmap(H0, [(energies[0],L,a,a2),(energies[1],L,a,a2),(energies[2],L,a,a2)])
    for i in prange(lenergies):
#        print(np.linalg.eig(result[i])[0])
#        res[i] = sorted(np.linalg.eig(result[i])[0])[-1].real  
        resaux = sorted(np.linalg.eig(result[i])[0])
#        res.append([energies[i], np.prod([resaux[0],resaux[2],resaux[4],resaux[6],resaux[8],resaux[10],resaux[12]]).real])   
        res.append( np.prod(resaux[0:4]).real)   
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
    result = pool.starmap(H0, aux)
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






  



L=28.47
print(L)
a=-0.1
a2=-1.3
Etest = 2.8785
Etest = 2.8786
dEtest = 0.0007


start = time.time()

energies= [Etest-dEtest, Etest, Etest+dEtest] # [3.000213837, 3.000216951408889, 3.000220065817778]
#energies = [3.000088646152035, 3.000088646154394, 3.000088646156753]
#energies = [3.0001409580067375, 3.0001409580069174, 3.0001409580070972]
#energies = [3.000140957589557, 3.0001409580070972, 3.0001409584246375]

#energies = [2.8777,2.878836242810217, 2.8794181214051084]

import matplotlib.pyplot as plt
#energies = [2.875864431675809, 2.875865261840274, 2.875866092004739]

#energies = [2.8745, 2.876, 2.8775]
#energies =[2.876192369385833, 2.8761943289205947, 2.8761962884553562]


for i in range(7):
    print(energies)
    res = EWrange2(L,a,a2,energies)
    print(res)
    resaux=res
    resaux=np.array(res)/max(res)

    print(resaux)
    
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
        exit()

    dEner = min([abs(energies[0]-Ener),abs(energies[2]-Ener)])

    energies[0] = Ener-dEner
    energies[2] = Ener+dEner
        
    print(energies)
        
        
end = time.time()
print('time is:', end - start, ' s')





