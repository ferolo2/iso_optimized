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
    
#3.258751036009

def F3i(e,L,a0,r,P,a2):

  K0 = 1000*5
  K1 = 3*270/9*0
  K2 = 40./81*0
  A = 40/81*0
  B = 400*0
  Kdf = K3.K3mat(e,L,K0,K1,K2,A,B)
  f3mat= F3_mat.F3i_mat(e,L,a0,r,P,a2,0.3)
  return pr.irrep_proj(f3mat + Kdf ,e,L,"T2+")

def F3i_00(e,L,a0,IPV):

#  K0=-1000*5*1.77315*0
  K0 = 248280.198
#  K0 = 61365.491
  K1 = 0
#  K1 = -144193.2
  
#  Kdf = K3.K3mat(e,L,K0,K1,K2,A,B)
  f3mat= F3_mat.F3imat00(e,L,a0,0.,0.,0.,0.3,IPV)
  #print(f3mat)
  Kdf = K0 +  (e**2-9)*K1/9  + f3mat*0
#  print('hola',pr.iso_proj00(f3mat,e,L))
#  return np.ndarray.flatten(pr.iso_proj00(f3mat,e,L))[0]
#  return np.linalg.eig(f3mat)
  return f3mat + Kdf



def K3pF3i(e,L,a0,a2):
    return F3i(e,L,a0,a2)+100000*K33Bmat(e,L,a0,a2)


  
def H2(e,L,a,a2):
    return H_mat.Hmat(e,L,a,0.,0.,a2,0.5)
def H2proj(e,L,a,a2):
    return pr.irrep_proj(H_mat.Hmat(e,L,a,0.,0.,a2,0.5),e,L,"A1+")
def H0(e,L,a,a2):
    return H_mat.Hmat00(e,L,a,0.,0.,0.5)

def EWrange2(L,a,r,P,a2,energies):
    res=np.zeros(len(energies))
    lenergies= len(list(energies))
    pool = Pool(processes=3)

    aux = []
    for i in range(lenergies):
        aux.append((energies[i],L,a,r,P,a2))

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
        resaux = sorted(np.linalg.eig(result[i])[0],key=abs)
#        res.append( np.prod([resaux[0],resaux[3],resaux[6],resaux[9],resaux[12],resaux[15],resaux[18],resaux[21]]).real)   T2p
        res.append( np.prod([resaux[0],resaux[3],resaux[6],resaux[9],resaux[12],resaux[15],resaux[18]]).real)   
#        res.append( np.prod([resaux[0],resaux[2],resaux[4]]).real)   
#        res.append( np.prod([resaux[0],resaux[2],resaux[4],resaux[6],resaux[8],resaux[10],resaux[12]]).real)   
#        res.append( np.prod(resaux[0:4]).real)
#        print(resaux[0:4])
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



def EWrange0(L,a0,IPV,energies):
    res=np.zeros(len(energies))
    lenergies= len(list(energies))
    pool = Pool(processes=4)

    aux = []
    for i in range(lenergies):
        aux.append((energies[i],L,a0,IPV))

    res=[]    
    result = pool.starmap(F3i_00, aux)
    for i in prange(lenergies):
      resaux = sorted(np.linalg.eig(result[i])[0],key=abs)
#      resaux = result[i]
      res.append( np.prod(resaux[0:1]).real)
    return res



  
def Ethres(L,a):
    I = -8.914
    J = I**2+16.532
    E3 = 12*math.pi*a/L**3*(1 - (a/math.pi/L)*I +(a/math.pi/L)**2*J )
    return E3



def Ethres(L,a):
    I = -8.914
    J = I**2+16.532
    E3 = 12*math.pi*a/L**3*(1 - (a/math.pi/L)*I +(a/math.pi/L)**2*J )
    return E3





L=19
a = 0.1
a2 = -0.1867
r = 56.21
P = -0.000308
IPV=0
#IPV=-1


#Etest = 2*np.sqrt(1+2*(2*math.pi/L)**2)+1+0.007
#Etest = 3+Ethres(L,a)
#Etest = 3.00014
Etest = 1+2*np.sqrt(1+(2*math.pi/L)**2)+8.8*Ethres(L,a)/3

dEtest = Ethres(L,a)*0.45
#dEtest = 0.0000001/3
#print('Efree = ', 2*np.sqrt(1+1*(2*math.pi/L)**2)+1)
start = time.time()

energies= [Etest-dEtest, Etest, Etest+dEtest] # [3.000213837, 3.000216951408889, 3.000220065817778]
print(Etest)

#import matplotlib.pyplot as plt
#energies = [3.466,3.468,3.47]
#energies=[4.6815, 4.6832, 4.684]


for i in range(7):
    print(energies)
    res = EWrange0(L,a,IPV,energies)
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
