#from scipy.special import sph_harm # no longer used
import math
import numpy as np
import sums_alt as sums
import F2_alt as F2
from defns import y2, y2real, list_nnk, lm_idx, chop, full_matrix
from numba import jit,njit

@jit(nopython=True,fastmath=True) #FRL, this speeds up like 5-10%
def npsqrt(x):
    return np.sqrt(x)
@jit(nopython=True,fastmath=True) #FRL, this speeds up like 5-10%
def square(x):
    return x**2

@jit(nopython=True,parallel=True,fastmath=True) #FRL this speeds up scalar prod of two vectors. DONT use for matrices.
def mydot(x,y):
    res = 0.
    for i in range(3):
        res+=x[i]*y[i]
    return res


def ktinvroot(e,L,nk):
    k = nk*2*math.pi/L
    omk = npsqrt(1+k**2)
    E2k = npsqrt(e**2 + 1 - 2*e*omk)
    return 1/npsqrt(32.*math.pi*omk*E2k )

def rho1mH(e, L, nk):
    k = nk*2*math.pi/L;
    hhk = sums.hh(e, k);
#    print(1-sums.E2a2(e,k))

    if hhk<1:
        return npsqrt(1-sums.E2a2(e,k))*(1-hhk)
    else:
        return 0.

@njit(fastmath=True,cache=True)
def boost(nnp, nnk,e):
    nnppar = np.array([0.,0.,0.])
    if(sums.norm(nnk)>0):
 #       nnppar = np.multiply(mydot(nnk,nnp)/sums.norm(nnk)**2, nnk)
        nnppar = nnk*(mydot(nnk,nnp)/sums.norm(nnk)**2)
    nnpperp = nnp - nnppar

    gamma = sums.gam(e, sums.norm(nnk))
    omp = npsqrt(1+square(sums.norm(nnp)))
    omk = npsqrt(1+square(sums.norm(nnk)))

    nnpparboost = np.multiply(nnppar, gamma) + np.multiply( np.multiply(nnk, gamma), (omp)/(e-omk ))


    return nnpparboost + nnpperp


# Calculate Gtilde = G/(2*omega)
# TB: Choose basis inside
#@njit(fastmath=True,cache=True)
@njit(fastmath=True,cache=True)
def G(e, L, nnp, nnk,l1,m1,l2,m2):
    p = sums.norm(nnp) * 2. *math.pi/L
    k = sums.norm(nnk) * 2. *math.pi/L
#    pk = sums.norm(np.add(nnk,nnp)) * 2. *math.pi/L
    pk = sums.norm(np.add(nnk,nnp)) * 2. *math.pi/L
    omp = npsqrt(1+square(p))
    omk = npsqrt(1+square(k))
    #ompk = np.sqrt(1+pk**2)

    bkp2 = (e-omp-omk)**2 - (2*math.pi/L)**2*sums.norm(np.add(nnk,nnp))**2
#    print('test')

    # nnps and nnks are the full vectors p* and k*
    nnps = boost(np.multiply(nnp, 2*math.pi/L), np.multiply(nnk, 2*math.pi/L), e)
    nnks = boost(np.multiply(nnk, 2*math.pi/L), np.multiply(nnp, 2*math.pi/L), e)
    #ps = sums.norm(nnps)
    #ks = sums.norm(nnks)
    qps2 = square(e - omp)/4 - square(p)/4 - 1
    qks2 = square(e - omk)/4 - square(k)/4 - 1

    # TB: Choose spherical harmonic basis
    Ytype = 'r' # 'r' for real, 'c' for complex
    Ylmlm = 1
    momfactor1 = 1
    momfactor2 = 1
    if(l1==2):
        #momfactor1 = (ks)**l1/qps2
        momfactor1 = qps2**(-l1/2) # TB: ks**l1 included in my y2(nnks)
        #Ylmlm = y2(nnks,m1,Ytype)
        Ylmlm = y2real(nnks,m1)
    if(l2==2):
        #momfactor2 = (ps)**l2/qks2
        momfactor2 = qks2**(-l2/2) # TB: ps**l2 included in my y2(nnps)
        #Ylmlm = Ylmlm * y2(nnps,m2,Ytype)
        Ylmlm = Ylmlm * y2real(nnps,m2)

    #out = sums.hh(e,p)*sums.hh(e,k)/(L**3 * 4*omp*omk*(bkp2-1)) *Ylmlm * momfactor1 * momfactor2
    out = sums.hh(e,p)*sums.hh(e,k)/(L**3 * 4*omp*omk*(bkp2-1)) *Ylmlm # TB, no q

    # if (Ytype=='r' or Ytype=='real') and abs(out.imag)>1e-15:
    #     sys.exit('Error in G: imaginary part in real basis output')
    return out.real


# Full Gtilde matrix (new structure)
def Gmat(E,L):
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
        [l1,m1] = lm_idx(i1)
        for i2 in range(6):
          [l2,m2] = lm_idx(i2)

          Gpk[i1,i2] = G(E,L,np.array(nnp),np.array(nnk),l1,m1,l2,m2)

      Gp.append(Gpk)

    Gfull.append(Gp)

  return chop(np.block(Gfull))


# Just compute l'=l=0 portion
@jit(fastmath=True,cache=True)
def Gmat00(E,L):
  nnk_list = list_nnk(E,L)
  N = len(nnk_list)
#  print(nnk_list)
#  print(list(nnk_list[0]))

  Gfull = np.zeros((N,N))
  for p in range(N):
#    nnp = list(nnk_list[p])
    nnp = nnk_list[p]
    for k in range(N):
#      nnk = list(nnk_list[k])
      nnk = nnk_list[k]
      Gfull[p,k] = G(E,L,np.array(nnp),np.array(nnk),0,0,0,0)

  return chop(Gfull)


# Just compute l'=l=2 portion
def Gmat22(E,L):
  nnk_list = list_nnk(E,L)
  N = len(nnk_list)

  Gfull = []
  for p in range(N):
    nnp = list(nnk_list[p])
    Gp = []
    for k in range(N):
      nnk = list(nnk_list[k])

      Gpk = np.zeros((5,5))
      for i in range(5):
        for j in range(5):
          mp = i-2;  m = j-2
          Gpk[i,j] = G(E,L,nnp,nnk,2,mp,2,m)
      Gp.append(Gpk)
    Gfull.append(Gp)
  return chop(np.block(Gfull))


# Just compute l'=l=0 portion in A1+ irrep (used to get iso part of H^-1)
def Gmat00_A1(E,L):
  shells = shell_list(E,L)
  Nshells = len(shells)

  out = np.zeros((Nshells,Nshells))
  # Each row/column corresponds to a different shell
  for ip in range(Nshells):
    op = shell_nnk_list(shells[ip])     # list of nnp's in shell op (index = ip)
    for ik in range(ip,Nshells):
      ok = shell_nnk_list(shells[ik])   # list of nnk's in shell ok (index = ik)

      # Sum over G(p,k) for p in op & k in ok (part of A1+ projection)
      for nnp in op:
        for nnk in ok:
          out[ip,ik] += G(E,L,nnp,nnk,0,0,0,0)
      out[ip,ik] *= 1/npsqrt(len(op)*len(ok))   # normalization factor

      if ip != ik:
        out[ik,ip] = out[ip,ik]   # take advantage of symmetry

  return chop(out)

######################################################################
# Old structure/Fernando code
######################################################################

# def PhiTheta(nnk):
#     x = nnk[0]
#     y = nnk[1]
#     z = nnk[2]

#     Theta = np.arctan2(np.sqrt(x**2+y**2),z)
#     Phi = np.arctan2(y,x)
#     if(Phi<0):
#         Phi = 2*math.pi+Phi

#     return  Phi, Theta


# Calculate G (this is really Gtilde = G/(2*omega))
# def G(e, L, nnp, nnk,l1,m1,l2,m2):

#     p = sums.norm(nnp) * 2. *math.pi/L
#     k = sums.norm(nnk) * 2. *math.pi/L
#     pk = sums.norm(np.add(nnk,nnp)) * 2. *math.pi/L
#     omp = np.sqrt(1+p**2)
#     omk = np.sqrt(1+k**2)
#     ompk = np.sqrt(1+pk**2)
#     bkp2 = (e-omp-omk)**2 - (2*math.pi/L)**2*sums.norm(np.add(nnk,nnp))**2
#     # nnps and nnks are the full vectors p* and k*
#     nnps = boost(np.multiply(nnp, 2*math.pi/L), np.multiply(nnk, 2*math.pi/L), e)
#     nnks = boost(np.multiply(nnk, 2*math.pi/L), np.multiply(nnp, 2*math.pi/L), e)
#     ps = sums.norm(nnps)
#     ks = sums.norm(nnks)
#     qps = (e - omp)**2/4 - p**2/4 - 1
#     qks = (e - omk)**2/4 - k**2/4 - 1

#     momfactor1 = 1
#     momfactor2 = 1
#     if(l1>1):
#         momfactor1 = (ks)**l1 * qps**(-l1/2)
#         #momfactor1 = (ks)**l1/qps
#         # shouldn't this be (ks/sqrt(qps))**l1 ? (fine if l1=2)
#     if(l2>1):
#         momfactor2 = (ps)**l2 * qks**(-l2/2)
#         #momfactor2 = (ps)**l2/qks
#         # shouldn't this be (ps/sqrt(qks))**l1 ? (fine if l2=2)

#     Phik, Thetak = PhiTheta(nnks)
#     Phip, Thetap = PhiTheta(nnps)

#     Ylmlm =4*math.pi* sph_harm(m1,l1,Phik,Thetak) * np.conj(sph_harm(m2,l2,Phip,Thetap))

# #    aux = (sums.hh(e,p)*sums.hh(e,k)/(L**3 * 4*omp*omk*(bkp2-1)) *Ylmlm * momfactor1 * momfactor2).imag
# #    print(aux)

#     return sums.hh(e,p)*sums.hh(e,k)/(L**3 * 4*omp*omk*(bkp2-1)) *Ylmlm * momfactor1 * momfactor2

def G00(e,L):

    nklist = list_nnk(e,L)
    G_00 = np.zeros((len(nklist),len(nklist)))
    #G_00 = np.zeros((len(nklist),len(nklist)),dtype=complex)

    i=0
    for nnp in nklist:  # TB: I reversed nnp and nnk
        j=0
        for nnk in nklist:
           # print(nnk, nnp)
            G_00[i][j]=G(e,L,nnp,nnk,0,0,0,0)
            j+=1
        i+=1

    return G_00


def G02m(e,L,m2):

    nklist = list_nnk(e,L)
    G_02 = np.zeros((len(nklist),len(nklist)))
    #G_02 = np.zeros((len(nklist),len(nklist)),dtype=complex)

    i=0
    for nnp in nklist:  # TB: I reversed nnp and nnk
        j=0
        for nnk in nklist:
           # print(nnk, nnp)
            G_02[i][j]=G(e,L,nnp,nnk,0,0,2,m2)
            j+=1
          #  print(G(e,L,nnp,nnk))
        i+=1

    return G_02


def G20m(e,L,m1):

    nklist = list_nnk(e,L)
    G_20 = np.zeros((len(nklist),len(nklist)))
    #G_20 = np.zeros((len(nklist),len(nklist)),dtype=complex)

    i=0
    for nnp in nklist: # TB: I reversed nnp and nnk
        j=0
        for nnk in nklist:
#            print(nnk, nnp)
            G_20[i][j]=G(e,L,nnp,nnk,2,m1,0,0)
            j+=1
        i+=1

    return G_20


def G22m(e,L,m1,m2):

    nklist = list_nnk(e,L)
    G_22 = np.zeros((len(nklist),len(nklist)))
    #G_22 = np.zeros((len(nklist),len(nklist)),dtype=complex)

    i=0
    for nnp in nklist: # TB: I reversed nnp and nnk
        j=0
        for nnk in nklist:
           # print(nnk, nnp)
            G_22[i][j]=G(e,L,nnp,nnk,2,m1,2,m2)
            j+=1
        i+=1

    return G_22


def G02(e,L):

    G_02 = G02m(e,L,-2)
    G_02 = np.hstack((G_02, G02m(e,L,-1)))
    G_02 = np.hstack((G_02, G02m(e,L,0)))
    G_02 = np.hstack((G_02, G02m(e,L,1)))
    G_02 = np.hstack((G_02, G02m(e,L,2)))
    return G_02

def G20(e,L):

    G_20 = G20m(e,L,-2)
    G_20 = np.vstack((G_20, G20m(e,L,-1)))
    G_20 = np.vstack((G_20, G20m(e,L,0)))
    G_20 = np.vstack((G_20, G20m(e,L,1)))
    G_20 = np.vstack((G_20, G20m(e,L,2)))
    return G_20

# TB: made edits
def G22(e,L):

    G22_m2 = G22m(e,L,-2,-2)
    G22_m2 = np.hstack((G22_m2, G22m(e,L,-2,-1)))
    G22_m2 = np.hstack((G22_m2, G22m(e,L,-2,0)))
    G22_m2 = np.hstack((G22_m2, G22m(e,L,-2,1)))
    G22_m2 = np.hstack((G22_m2, G22m(e,L,-2,2)))

    G22_m1 = G22m(e,L,-1,-2)
    G22_m1 = np.hstack((G22_m1, G22m(e,L,-1,-1)))
    G22_m1 = np.hstack((G22_m1, G22m(e,L,-1,0)))
    G22_m1 = np.hstack((G22_m1, G22m(e,L,-1,1)))
    G22_m1 = np.hstack((G22_m1, G22m(e,L,-1,2)))

    G22_0 = G22m(e,L,0,-2)
    G22_0 = np.hstack((G22_0, G22m(e,L,0,-1)))
    G22_0 = np.hstack((G22_0, G22m(e,L,0,0)))
    G22_0 = np.hstack((G22_0, G22m(e,L,0,1)))
    G22_0 = np.hstack((G22_0, G22m(e,L,0,2)))

    G22_p1 = G22m(e,L,1,-2)
    G22_p1 = np.hstack((G22_p1, G22m(e,L,1,-1)))
    G22_p1 = np.hstack((G22_p1, G22m(e,L,1,0)))
    G22_p1 = np.hstack((G22_p1, G22m(e,L,1,1)))
    G22_p1 = np.hstack((G22_p1, G22m(e,L,1,2)))

    G22_p2 = G22m(e,L,2,-2)
    G22_p2 = np.hstack((G22_p2, G22m(e,L,2,-1)))
    G22_p2 = np.hstack((G22_p2, G22m(e,L,2,0)))
    G22_p2 = np.hstack((G22_p2, G22m(e,L,2,1)))
    G22_p2 = np.hstack((G22_p2, G22m(e,L,2,2)))

    G_22 = np.vstack((G22_m2, G22_m1, G22_0, G22_p1, G22_p2))

    return G_22

#full Gtilde matrix (old structure)
def Gmat_old(E,L):
  return full_matrix( G00(E,L), G20(E,L), G02(E,L), G22(E,L) )


##########################################################
# TB: I haven't touched anything below here (other than commenting out the last portion)

# Create -1/a matrix in eq.(A1) of isotropic paper
def A00(e,L,a):

    nklist = list_nnk(e,L)
    matrixa =  np.diag(np.ones(len(nklist))*(-1/a))
    return matrixa


def A22(e,L,a2):

    nklist = list_nnk(e,L)
#    q = np.sqrt(1-sums.E2a2(e,k))
#    nklist = list_nnk(e,L)
#    matrixa =  np.diag(np.ones(len(nklist)*5)*(-1/(a2**5*q**4)))
    res = []

    for i in range(5):
        for nnk in nklist:
            q4 = (1-sums.E2a2(e,2*math.pi*sums.norm(nnk)/L))**4
            res.append(-1/(a2**5*q4))
 #   print(res)
    return np.diag(res)


def RHO00(e,L):

    nklist = list_nnk(e,L)
    res = []
    for nnk in nklist:
        res.append(rho1mH(e,L,sums.norm(nnk)))

    return np.diag(res)

def RHO22(e,L):

    nklist = list_nnk(e,L)
    res = []
    for i in range(5):
        for nnk in nklist:
            res.append(rho1mH(e,L,sums.norm(nnk)))

    return np.diag(res)


def Xi00(e,L):
    nklist = list_nnk(e,L)
    res = []
    for nnk in nklist:
        res.append(1./ktinvroot(e,L,sums.norm(nnk)))

    return np.diag(res)


def Xi22(e,L):
    nklist = list_nnk(e,L)
    res = []
    for i in range(5):
        for nnk in nklist:
            res.append(1./ktinvroot(e,L,sums.norm(nnk)))

    return np.diag(res)


def detF2(e,L,a):

    xi00 = Xi00(e,L)
    g00 = G00(e,L)
    rho00 = RHO00(e,L)
    a00 = A00(e,L,0.1)
    f00 = F2.full_F2_00_matrix(e,L)

    matrix1 = np.dot(xi00, f00+g00)
    matrix2 = rho00 + np.dot(matrix1,xi00) + a00

    detres = np.linalg.det(matrix2)
    return detres


def detF2_dwave(e,L,a,a2):
    XI = np.diag(np.concatenate( (np.diag(Xi00(e,L)),np.diag(Xi22(e,L)))) )
    RHO =  np.diag(np.concatenate( (np.diag(RHO00(e,L)),np.diag(RHO22(e,L)))) )
    A = np.diag(np.concatenate( (np.diag(A00(e,L,a)),np.diag(A22(e,L,a2)))) )
    G = full_matrix(G00(e,L),G20(e,L),G02(e,L),G22(e,L))

    f00 = F2.full_F2_00_matrix(e,L)
    f20 = F2.full_F2_20_matrix(e,L)
    f02 = F2.full_F2_02_matrix(e,L)
    f22 = F2.full_F2_22_matrix(e,L)

    F = full_matrix(f00,f20,f02,f22)

    matrix1 = np.dot(XI, F+G)
    matrix2 = RHO + np.dot(matrix1,XI) + A

    detres = np.linalg.det(matrix2)
    return detres



def frange(start, stop, step=1.0):
    ''' "range()" like function which accept float type'''
    i = start
    while i < stop:
        yield i
        i += step


#import time

# L=5
# a=0.1
# for i in frange(3.0315,3.0321,0.0001):
#     start = time.time()
#     print('det F2 at E=',i,' is ' ,detF2(i,L,a))
#     print('det F2 at E=',i,' is ' ,detF2_dwave(i,L,a,a))
#     end = time.time()
#     print('time is:', end - start, ' s')
