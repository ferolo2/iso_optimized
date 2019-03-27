import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg
from matplotlib import pyplot as plt
from ast import literal_eval

from scipy.linalg import eig

import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
sys.path.insert(0,cwd+'/K3df')
#sys.path.insert(0,cwd)

from K3df import K3A, K3B, K3quad
from F3 import F2_alt, Gmatrix, sums_alt as sums, K2i_mat
from F3 import H_mat, F3_mat
import defns, projections as proj, analysis_defns as AD, group_theory_defns as GT

####################################################################################
# Playground file to test functions
# Note: input order for all matrix elements is (E,outgoing momenta,incoming momenta)
####################################################################################
# Define parameters (necessary for several functions)

# K2 parameters
# a0=-10; r0=0.5; P0=0.5; a2=-1
# data_dir = 'Data/a0=m10_r0=0.5_P0=0.5_a2=m1/'

# a0=0.1; r0=0; P0=0; a2=1
# data_dir = 'Data/a0=0.1_r0=0_P0=0_a2=1/'

a0=0.1; r0=0; P0=0; a2=0.5
data_dir = 'Data/a0=0.1_r0=0_P0=0_a2=0.5/'


# F2_KSS parameter
alpha=0.5

# K3df parameters
K0=0; K1=0; K2=0; A=1; B=1
# Spherical harmonic basis (currently only for K3df functions)
Ytype='r'  # 'r' for real, 'c' for complex

# Total CM energy; lattice size
E_list = [3.04]
L_list = [5.0]

# E_list = np.arange(2.95,3.81,0.05)
# E_list = [round(E,2) for E in E_list]
# E_list.remove(3.0)


# print(sqrt(8*pi**2/defns.kmax(2.95)**2))
# print(sqrt(8*pi**2/defns.kmax(3.12)**2))

# print('E='+str(E)+', L='+str(L)+'\n')
# print('Shells: ', defns.shell_list(E,L), '\n')
# print('Matrix dimension: ', len(defns.list_nnk(E,L))*6, '\n')



####################################################################################
# Load F3 matrix from file if exists, otherwise compute from scratch & save to file
# E_list = [3.97]
# L_list = [5.0]
# for L in L_list:
#   F3_list=[]
#   for E in E_list:
#     datafile = data_dir+'F3_E'+str(E)+'_L'+str(L)+'.dat'

#     try:
#       with open(datafile, 'rb') as fp:
#         F3 = pickle.load(fp)
#         print('F3 loaded from '+datafile+'\n')

#     except IOError:
#       print(datafile+' not found; computing F3 from scratch...')
#       t0 = time.time()
#       F3 = F3_mat.F3mat(E,L,a0,r0,P0,a2,alpha) 
#       t1=time.time()
#       print('Calculation complete (time:',t1-t0,')')
#       with open(datafile, 'wb') as fp:
#         pickle.dump(F3, fp)
#         print('F3 saved to '+datafile+'\n')

    #F3_list.append( F3 )


####################################################################################
# Eigenvalue analysis
#E=3.1; L=5.0
#a0=0.1; r0=0; P0=0; a2=1
#I = 'T2-'
## print(defns.shell_list(E,L))
#
#P = proj.P_irrep_full(E,L,I)
#Psub = proj.P_irrep_subspace(E,L,I)
#M = F3_mat.F3mat(E,L,a0,r0,P0,a2,alpha)
#Mi = defns.chop(LA.inv(M))
##M = Gmatrix.Gmat(E,L)
#
#eigs = sorted(defns.chop(LA.eigvals(Mi)))
#eigs1 = sorted(defns.chop(LA.eigvals(defns.chop(P@Mi@P)).real))
#eigs2 = sorted(LA.eigvals(defns.chop(Psub.T@Mi@Psub)).real)
#
##eigs3 = sorted(defns.chop(LA.eigvals(defns.chop(LA.inv(P@M@P)))))
#eigs4 = sorted(LA.eigvals(defns.chop(LA.inv(Psub.T@M@Psub))).real)
#
#print([e for e in eigs1 if abs(e)>1e-10],'\n')
#print(eigs2,'\n')
##print([e for e in eigs3 if abs(e)>0],'\n')
#print(eigs4,'\n')
#
#print([e for e in eigs if round(e,4) in [round(i,4) for i in eigs2]])

###########################
# Check projection matrix diagonalization
# S = np.zeros((114,0))
# for I in GT.irrep_list():
#   if I!='A1-':
#     S = np.concatenate((S,proj.P_irrep_subspace(E,L,I)),axis=1)
# P = proj.P_irrep_full(E,L,'T1+')
# print(np.diag(defns.chop(S.T@P@S)))


# free_list = defns.E_2pt_free_list(L,4,2)
# for d in free_list:
#   E2 = free_list[d]
#   e_list=[]
#   for nk in defns.shell_list(E,L):
#     k=LA.norm(nk)*2*pi/L
#     e_list.append(round(defns.omega(k)+sqrt(k**2+E2**2),10))
#   print(d,round(E2,4),e_list)
    

# #First few 3-pt. free energies
# free_list = defns.E_free_list(L,10,3)
# for d in free_list:
#   print(d,'--',round(free_list[d],6))


# Energies where shells open up
# for shell in defns.shell_list(10,L):
#   print(shell,defns.Emin(shell,L))


# F3_I = proj.irrep_proj(F3,E,L,I)
# F3i_I = defns.chop(LA.inv(F3_I))
# w_list,v_list = LA.eig(F3i_I)

# F3i = defns.chop(LA.inv(F3))
# w_list,v_list = LA.eig(F3i)
# w_list = w_list.real; v_list = v_list.real
# for v in v_list:
#   proj.evec_decomp(v,E,L,I)

# w0,v0 = AD.smallest_eig_evec(F3i)
# w0 = w0.real; v0 = v0.real
# print(w0)
# #proj.evec_decomp(v0,E,L,I)

# E=4.2; L=5
#
# v0 = np.random.rand(114,)*10
# s = 0
# for I in GT.irrep_list():
#   s += proj.evec_decomp(v0,E,L,I)
# print('Total:',s)




# E=3.96
# print(xx2_TB(E,5,[0,0,1]))
# print(rr2_TB(E,5,[0,0,1],[0,0,0]))


import roots_master
#roots_master.main()


####################################################################################
# Compute little group sums of Wigner-D matrices (needed for A1 projection)

#print(proj.A1_little_group_sum([1,2,0]))


# F3i = defns.chop(LA.inv(F3))
# #F3_00 = proj.l0_proj(F3)
# F3_00 = F3_mat.F3mat00(E,L,a0,r0,P0,a2,alpha)
# F3i_00 = defns.chop(LA.inv(F3_00))

#print(F3i_00 - proj.l0_proj(F3i))
# above is non-zero, so must do l=0 projection before inverting




#print( p_iso_00.T @ F3i_00 @ p_iso_00 )
#print( defns.chop(LA.inv(p_iso_00.T @ F3_00 @ p_iso_00)))
# results are identical, so order of A1 projection and inversion is irrelevant (provided l=0 done before both)



# print(LA.eigvals(p_iso_00.T @ F3i_00 @ p_iso_00))

# p_iso_ones = np.ones((7,1))
# print( p_iso_ones.T @ F3i_00 @ p_iso_ones )
# print( defns.chop(LA.inv(p_iso_ones.T @ F3_00 @ p_iso_ones)))



# p_iso = proj.p_A1(E,L)[:,0:2]

# F3_iso = p_iso.T @ F3 @ p_iso
#print(p_iso.T @ F3i @ p_iso)
#print(defns.chop(LA.inv(F3_iso)))


  

# F3i = defns.chop(LA.inv(F3))
# P_A1 = proj.P_A1_subspace(E,L)
# #print(np.trace(P_A1))

# new_eigs = sorted(defns.chop(LA.eigvals(defns.chop(P_A1.T @ F3i @ P_A1)).real))

# old_eigs = sorted(defns.chop(LA.eigvals(defns.chop(LA.inv(proj.A1_proj_old(F3,E,L)))).real))
# print(old_eigs)
# print(new_eigs)


####################################################################################
# Load roots and compare

# E_list = np.arange(3.02,3.11,0.02)
# E_list = [round(E,2) for E in E_list]

# L_list = np.arange(5.0,6.41,0.2)
# L_list = [round(L,1) for L in L_list]

# data_dir = 'Data/a0=0.1_r0=0_P0=0_a2=1/'

# L_list = np.arange(5.0,6.21,0.2)
# L_list = [round(L,1) for L in L_list]

# A1_root_list = []
# iso_root_list = []
# diff_list = []
# for L in L_list:
#   A1_file = data_dir+'A1_roots_L='+str(L)+'.dat'
#   iso_file = data_dir+'iso_roots_L='+str(L)+'.dat'

#   with open(A1_file,'r') as f_A1:
#     A1_root = literal_eval(f_A1.read())[0]
#     A1_root_list.append(A1_root)
  
#   with open(iso_file,'r') as f_iso:
#     iso_root = literal_eval(f_iso.read())[0]
#     iso_root_list.append(iso_root)

#   diff_list.append(A1_root-iso_root)

# print('iso roots:\n',list(zip(L_list,iso_root_list)),'\n')
# print('A1 roots:\n',list(zip(L_list,A1_root_list)),'\n')
# print('difference:\n',list(zip(L_list,diff_list)))

# plt.plot(L_list,diff_list,'.')
# plt.xlabel('L'); plt.ylabel(r'$E_{A_1^+}-E_{iso}$')
# plt.tight_layout(); plt.grid(True)
# # plt.ylim((-1e2,1e2))
# plt.savefig('A1_iso_comparison.pdf')


####################################################################################
# Test F3 code

#t0 = time.time()
# E = 3.96798890
# Fmat = F2_alt.Fmat(E,L,alpha); #print('F2: \n', Fmat)
# #Fmat_eigs = np.array(sorted(LA.eigvals(Fmat),reverse=True)).real; print(Fmat_eigs)
# Fmati = defns.chop(LA.inv(Fmat))
# Fmati_eigs = np.array(sorted(LA.eigvals(Fmati),reverse=True)).real; print(Fmati_eigs)

#Gmat = Gmatrix.Gmat(E,L); #print('G: \n', Gmat)
#Gmat_eigs = np.array(sorted(LA.eigvals(Gmat),reverse=True)).real; print(Gmat_eigs)

#Hmat = H_mat.Hmat(E,L,a0,r0,P0,a2); #print('H: \n', Hmat)
#Hmat_eigs = np.array(sorted(LA.eigvals(Hmat),reverse=True)).real; print(Gmat_eigs)

#F3 = F3_mat.F3mat(E,L,a0,r0,P0,a2,alpha); #print('F3: \n', F3)
#F3i = F3_mat.F3i_mat(E,L,a0,r0,P0,a2,alpha); #print('F3i: \n', F3i)

# F3i_eigs = np.array(sorted(LA.eigvals(F3i),reverse=True)).real; print(F3i_eigs)

#t1 = time.time(); print('time: ',t1-t0)


####################################################################################
# Check individual matrix elements

nnp=[0,0,0]; nnk=[0,0,0]
lp=2; mp=0; l=2; m=0

#pvec=[i*2*pi/L for i in nnp]; kvec=[i*2*pi/L for i in nnk]

# print('F2:', sums.F2KSS(E,L,nnk,lp,mp,l,m,alpha))
# print('G:', Gmatrix.G(E,L,nnp,nnk,lp,mp,l,m))
# print('K2i:', H_mat.K2inv(E,kvec,l,m,a0,r0,P0,a2))
# print('H:', H_mat.H(E,L,nnp,lp,mp,nnk,l,m,a0,r0,P0,a2,alpha))

####################################################################################
# Test K3df code (see K3df/master.py)

# from K3df import master

####################################################################################
# Test detF2 and detF2_dwave in Gmatrix.py (copied from Fernando's Gmatrix.py)

# import time                                                        
# L=5
# a=0.1
# for i in Gmatrix.frange(3.0315,3.0321,0.0001):
#     start = time.time()                                                                                                                                                  
#     print('det F2 at E=',i,' is ' ,Gmatrix.detF2(i,L,a))
#     print('det F2 at E=',i,' is ' ,Gmatrix.detF2_dwave(i,L,a,a))
#     end = time.time()                                                                                                                                                                  
#     print('time is:', end - start, ' s')
