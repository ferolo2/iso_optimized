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
    f3mat= F3_mat.F3i_mat(e,L,a0,0.,0.,a2,0.5)
    return pr.irrep_proj(f3mat,e,L,"E+")

def F3iK(e,L,a0,a2):
  r=0.
  P=0.
  K0 = 100*0
  K1 = 10*0
  K2 = 200./81*-1* 0
  A = 200/81*0
  B = 40/81*-20000*0
  Kdf = K3.K3mat(e,L,K0,K1,K2,A,B)
  f3mat= F3_mat.F3i_mat(e,L,a0,r,P,a2,0.3)
  return pr.irrep_proj(f3mat + Kdf ,e,L,"E+")

def K3pF3i(e,L,a0,a2):
    return F3i(e,L,a0,a2)+100000*K33Bmat(e,L,a0,a2)


  
def H2(e,L,a,a2):
    return H_mat.Hmat(e,L,a,0.,0.,a2,0.5)
def H2proj(e,L,a,a2):
    return pr.irrep_proj(H_mat.Hmat(e,L,a,0.,0.,a2,0.5),e,L,"E+")
def H0(e,L,a,a2):
    return H_mat.Hmat00(e,L,a,0.,0.,0.5)

def EWrange2(L,a,a2,energies):
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
    result = pool.starmap(F3iK, aux)
#    result = map(K33Bmat, aux)
#    result = pool.starmap(H0, [(energies[0],L,a,a2),(energies[1],L,a,a2),(energies[2],L,a,a2)])
    for i in prange(lenergies):
#        print(np.linalg.eig(result[i])[0])
#        res[i] = sorted(np.linalg.eig(result[i])[0])[-1].real  
        resaux = sorted(np.linalg.eig(result[i])[0],key=abs)
        res.append([energies[i], np.prod([resaux[0],resaux[2],resaux[4],resaux[6]]).real])   
#        res.append([energies[i], np.prod(resaux[0:5]).real])   
#        res.append([energies[i],resaux[0].real,resaux[1].real,resaux[2].real,resaux[3].real,resaux[4].real,resaux[5].real])
#        res.append([energies[i],resaux[0].real,resaux[1].real,resaux[2].real,resaux[3].real,resaux[4].real,resaux[5].real,resaux[6].real,resaux[7].real,resaux[8].real,resaux[9].real])
#        res.append([energies[i], resaux[0].real])
        print(energies[i],resaux[0].real,resaux[2].real,resaux[4].real,resaux[6].real,resaux[8].real,resaux[10].real)
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
#        res.append([energies[i], np.prod(resaux).real])

        print(resaux)
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






  



L=10
#L=10
a=-0.1
a2=-0.1
#Etest = 3+Ethres(L,a)
#dEtest = Ethres(L,a)*0.2
#Etest = 2*np.sqrt(1+(2*math.pi/L)**2)+1-0.00004
Etest = 2*np.sqrt(1+2*(2*math.pi/L)**2)+1-0.038
#Etest = 2.986
#Etest = 2.741
#Etest = 4.00591
Etest = 3.50 
dEtest = 0.00125*2
#dEtest = 0.0
nmax=40
#Etest = 3.000059325557361  + 1e-12
#dEtest = 2e-12




start = time.time()

energies= [Etest-dEtest, Etest, Etest+dEtest] # [3.000213837, 3.000216951408889, 3.000220065817778]
#energies = [3.000088646152035, 3.000088646154394, 3.000088646156753]
#energies = [3.0001409580067375, 3.0001409580069174, 3.0001409580070972]
#energies = [3.000140957589557, 3.0001409580070972, 3.0001409584246375]


energies = []
for i in range(0,nmax):
    energies.append(Etest+i*dEtest)


#data = np.array(EWrange2(L,a,a2,energies))

#EN = data[:,0]
#EV1 = data[:,1]
#EV2 = data[:,2]
#EV3 = data[:,3]
#EV4 = data[:,4]
#EV5 = data[:,5]
#EV6 = data[:,6]
#EV7 = data[:,7]
#EV8 = data[:,8]
#EV9 = data[:,9]
#EV10 = data[:,10]

#EV=data[:,1]

xnew = np.linspace(Etest, Etest + (nmax-1)*dEtest, num=100, endpoint=True)
#f = interp1d(EN, EV1)
#f2 = interp1d(EN, EV1, kind='quadratic')
#f3 = interp1d(EN, EV1, kind='cubic')
#fb = interp1d(EN, EV2)
#f2b = interp1d(EN, EV2, kind='quadratic')
#f3b = interp1d(EN, EV,kind='cubic')
#f4b = interp1d(EN, EV2, kind=5)
#f5b = interp1d(EN, EV2, kind=7)

import matplotlib.pyplot as plt
#plt.plot(EN, EV1, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--',EN,EV2,'o',xnew, fb(xnew), '-', xnew, f2b(xnew), '--')
#plt.plot(EN,EV1,'o')
#plt.plot(EN,EV2,'x')
#plt.plot(EN,EV3,'s')
#plt.plot(EN,EV4)
#plt.plot(EN,EV5)
#plt.plot(EN,EV6)
#plt.plot(EN,EV7)
#plt.plot(EN,EV8)
#plt.plot(EN,EV9)
#plt.plot(EN,EV,'o',xnew, f3b(xnew), '-')


#plt.legend(['data', 'linear', 'cubic','data2', 'linear2', 'cubic2'], loc='best')

#print('sol from linear E = %.12f' % fsolve(fb,Etest)[0],'other sols',fsolve(fb,Etest))
#print('sol from quad E = %.12f' % fsolve(f2b,Etest)[0],'other sols',fsolve(f2b,Etest))
#print('sol from cubic E = %.12f' % fsolve(f3b,Etest+dEtest*10)[0],'other sols',fsolve(f3b,Etest))
#print('sol from 5th E = %.12f' % fsolve(f4b,Etest)[0],'other sols',fsolve(f4b,Etest))
#print('sol from 7th E = %.12f' % fsolve(f5b,Etest)[0],'other sols',fsolve(f5b,Etest))


print('free energy at %.12f' % (2.*np.sqrt(1+(2*math.pi/L)**2)+1.))
print('Leading threshold expansion %.12f' % (2*np.sqrt(1+(2*math.pi/L)**2)+1 + 16*a*math.pi/L**3))

#print(Etest)
#print(energies)

#data = np.array(EWrange2(L,a,-0.25,energies))
#print(data)
#EN = data[:,0]
#EV10 =data[:,1]
#print(data)

#f3b = interp1d(EN, EV10, kind='quadratic') 
#print('a2 = -0.01','sol from cubic E = %.12f' % fsolve(f3b,Etest+dEtest*20)[0])

rangea2=[-0.1,-0.2,-0.3,-0.45,-0.5,-0.6,-0.74,-0.76,-1.0,-1.335,-1.33,-1.33,-1.32] #rangea2=[-0.1,-0.2,-0.3]
rangea2=[-1.005,-1.02,-1.1,-1.15,-1.2,-1.335,-1.33,-1.33,-1.32] #rangea2=[-0.1,-0.2,-0.3]#
#rangea2=[-0.71] #,-0.97,-0.98,-0.99,-1,-1.01,-1.02,-1.03,-1.08,-1.09,-1.1,-1.15,-1.2,-1.335,-1.33,-1.33,-1.32] #rangea2=[-0.1,-0.2,-0.3]
#rangea2=[-0.5,-0.60,-0.625,-0.65,-0.67,-0.70,-0.74,-0.76,-1.335,-1.33,-1.33,-1.32] #rangea2=[-0.1,-0.2,-0.3]
#rangea2=[-0.67,-0.70,-0.74,-0.76,-1.335,-1.33,-1.33,-1.32] #rangea2=[-0.1,-0.2,-0.3]
rangea2=[0.1,0.2,0.4,0.6,0.7,0.8,0.9,1.0] #rangea2=[-0.1,-0.2,-0.3]
#rangea2=[-1.35,-1.3,-1.26,-1.22,-1.18,-1.10,-1.05,-1.04,-1.03,-1.02]
rangea2=[0.9999]#,0.9219,0.9220,0.9221,0.9222,0.9223,0.9224,0.9225]
#rangea2=[-1.85,-1.8,-1.75,-1.7,-1.65,-1.6]
#rangea2=[-0.5,-0.6,-0.7,-0.8,-1,-1.18,-1.22,-1.26,-1.3,-1.35,-1.4,-1.45,-1.5,-1.55,-1.6,-1.7,-1.8,-2] #rangea2=[-0.1,-0.2,-0.3]
#rangea2=[-1.1,-1.18,-1.22,-1.26,-1.3,-1.35,-1.4,-1.45,-1.5,-1.55,-1.6,-1.7,-1.8,-2] #rangea2=[-0.1,-0.2,-0.3]


#rangea2=[-2]

sols = []

aux=Etest

#aux =2.79
#aux= 2.7
for a2 in rangea2:

  energies = []
  for i in range(0,nmax):
    energies.append(aux+i*dEtest)


  data = np.array(EWrange2(L,a,a2,energies))

  print(data)

  EN = data[:,0]
  EV10 =data[:,1]

  f3b = interp1d(EN, EV10, kind='cubic') 
  
#  print(EN,EV10)
#  sols.append(fsolve(f3b,aux+dEtest*10)[0])
#  print('a2 = %f' % a2,'sol from cubic E = %.12f' % fsolve(f3b,aux+dEtest*10)[0])
#  aux=fsolve(f3b,aux+dEtest*10)[0]


  #  if(a2>-1.25):
#    dEtest = 0.00131
#  if(a2<-1.2):
#    aux=2.7
#  aux=fsolve(f3b,aux+dEtest*15)[0]
#  plt.show()

print(rangea2,sols)

plt.plot(rangea2,sols,'o')

plt.show()
exit()


data = np.array(EWrange2(L,a,5*a2,energies))
EV10 = data[:,1]
f3b = interp1d(EN, EV10, kind='cubic')
plt.plot(EN,EV10,'o',xnew, f3b(xnew), '-')
print('sol from cubic E = %.12f' % fsolve(f3b,Etest)[0],'other sols',fsolve(f3b,Etest))
data = np.array(EWrange2(L,a,-2*a2,energies))
EV10 = data[:,1]
f3b = interp1d(EN, EV10, kind='cubic')
plt.plot(EN,EV10,'o',xnew, f3b(xnew), '-')
print('sol from cubic E = %.12f' % fsolve(f3b,Etest)[0],'other sols',fsolve(f3b,Etest))
data = np.array(EWrange2(L,a,-3*a2,energies))
EV10 = data[:,1]
f3b = interp1d(EN, EV10, kind='cubic')
plt.plot(EN,EV10,'o',xnew, f3b(xnew), '-')
print('sol from cubic E = %.12f' % fsolve(f3b,Etest)[0],'other sols',fsolve(f3b,Etest))






print(Ethres(L,a))



for i in range(2):
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




