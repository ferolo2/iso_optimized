import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg

from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import bisect, newton
import sys

import defns, projections as proj

#################################################################
# Here we define functions that take lists of F3 matrices as input to find projected eigenvalues, eigenvalue roots, etc.
#################################################################

# Create list of eigenvalues of F3i (contains both l=0 and l=2)
def F3i_eigs_list( E_list, L, F3_list ):
  out = []
  for i in range(len(E_list)):
    F3 = F3_list[i]

    F3i = defns.chop(LA.inv(F3))
    F3i_eigs = np.array(sorted(LA.eigvals(F3i))).real

    out.append( F3i_eigs )
  return out


# Create list of eigenvalues of F3i in irrep I (contains both l=0 and l=2)
def F3i_I_eigs_list( E_list, L, F3_list, I, flip=False ):
  out = []
  for i in range(len(E_list)):
    E = E_list[i]
    F3 = F3_list[i]
    
    if flip==False: # project before inverting F3
      F3_I = proj.irrep_proj(F3,E,L,I) # works for all shells
      if F3_I.shape==():
        F3i_I = 1/F3_I
        F3i_I_eigs = F3i_I
      else:
        F3_I_eigs = defns.chop(np.array(sorted(LA.eigvals(F3_I),key=abs,reverse=True)).real)
        #F3i_I = defns.chop(LA.inv(F3_I))
        if F3_I_eigs[-1]==0:
          print(E,F3_I_eigs)
          sys.exit()
        else:
          F3i_I_eigs = np.array([1/x for x in F3_I_eigs])

    elif flip==True: # invert F3 before projecting
      F3i = defns.chop(LA.inv(F3))
      F3i_I = proj.irrep_proj(F3i,E,L,I)
      F3i_I_eigs = defns.chop(np.array(sorted(LA.eigvals(F3i_I))).real)
      
    out.append( F3i_I_eigs )
  return out


# Create list of eigenvalues of F3i_00 in irrep I
def F3i_00_I_eigs_list( E_list, L, F3_00_list, I, flip=False ):
  out = []
  for i in range(len(E_list)):
    E = E_list[i]
    F3 = F3_00_list[i]
    
    if flip==False: # project before inverting F3
      F3_I = proj.irrep_proj_00(F3,E,L,I) # works for all shells
      if F3_I.shape==():
        F3i_I = 1/F3_I
        F3i_I_eigs = F3i_I
      else:
        F3i_I = defns.chop(LA.inv(F3_I))

    elif flip==True: # invert F3 before projecting
      F3i = defns.chop(LA.inv(F3))
      F3i_I = proj.irrep_proj_00(F3i,E,L,I)

    F3i_I_eigs = defns.chop(np.array(sorted(LA.eigvals(F3i_I))).real)
      
    out.append( F3i_I_eigs )
  return out


# Create list of eigenvalues of F3i_22 in irrep I
def F3i_22_I_eigs_list( E_list, L, F3_22_list, I, flip=False ):
  out = []
  for i in range(len(E_list)):
    E = E_list[i]
    F3 = F3_22_list[i]
    
    if flip==False: # project before inverting F3
      F3_I = proj.irrep_proj_22(F3,E,L,I) # works for all shells
      if F3_I.shape==():
        F3i_I = 1/F3_I
        F3i_I_eigs = F3i_I
      else:
        F3i_I = defns.chop(LA.inv(F3_I))

    elif flip==True: # invert F3 before projecting
      F3i = defns.chop(LA.inv(F3))
      F3i_I = proj.irrep_proj_22(F3i,E,L,I)

    F3i_I_eigs = defns.chop(np.array(sorted(LA.eigvals(F3i_I))).real)
      
    out.append( F3i_I_eigs )
  return out

######################################
# Pick out eigenvalues closest to zero
def small_eig_list( eig_array_list ):
  small_list = []
  for i in range(len(eig_array_list)):
    eig_array = eig_array_list[i]

    small_eig = eig_array[0]
    for eig in eig_array:
      if abs(eig) < abs(small_eig):
        small_eig = eig
    small_list.append(small_eig)

  return small_list


# Pick out smallest eigenvalue & corresponding eigenvector
def smallest_eig_evec( M ):
  eigs, vecs = LA.eig(M)
  i0 = min(range(len(eigs)), key=lambda j: abs(eigs[j]))
  return (eigs[i0],vecs[i0])

###################################################
# Root-finding

# Find roots using cubic-spline interpolation (absolute tolerance 1e-8)
def spline_roots(E_list,eig_list,kind='cubic'):
  eig_spline = interp1d(E_list,eig_list,kind)

  Emin_list = []
  Emax_list = []
  for i in range(len(E_list)-1):
    if eig_list[i]*eig_list[i+1] <= 0:
      Emin_list.append(E_list[i])
      Emax_list.append(E_list[i+1])

  roots_list = []
  for i in range(len(Emin_list)):    
    root = bisect(eig_spline,Emin_list[i],Emax_list[i],xtol=1e-8)
    roots_list.append(root)

  return roots_list


# More precise root-finder
def root_finder(E_list, eig_array_list, f_eigs, inputs):
  small_eigs = small_eig_list(eig_array_list)
  roots_list = spline_roots(E_list,small_eigs)
  
  E_list_new = E_list
  final_roots_list = roots_list
  for i in range(len(roots_list)):
    E = roots_list[i]
    
    i0 = min(range(len(E_list)), key=lambda j: abs(E_list[j]-E))
    if E_list[i0]<E:
      Emin = E_list[i0]
      Emax = E_list[i0+1]
      i0 += 1   # so that Emax = E_list[i0] (for below)
    elif E_list[i0]>E:
      Emin = E_list[i0-1]
      Emax = E_list[i0]
    else:
      Emin=E; Emax=E
    
    print('Root in range',[Emin,Emax],', next guess: E = ',E)
    
    # Iteratively hone in on root
    count = 0
    while Emax-Emin>1e-8:
      test_eigs = f_eigs(E,*inputs)
      # test_mat = f_mat(E,*inputs)
      # if test_mat.shape == ():
      #   test_eigs = [test_mat]
      # else:
      #   test_eigs = np.array(sorted(LA.eigvals(test_mat))).real

      smallest = test_eigs[0]
      for eig in test_eigs:
        if abs(eig)<abs(smallest):
          smallest = eig
      
      # Assumes eigenvalue is increasing through 0
      if smallest<0:
        Emin = E
        i0 += 1  # so E_list[i0] = Emax
      elif smallest>0:
        Emax = E
      else:
        break

      E_list_new.insert(i0,E)
      small_eigs.insert(i0,smallest)
      
      # Use spline_roots to get next guess for E
      new_roots = spline_roots(E_list_new,small_eigs)
      E = [e for e in new_roots if Emin<e<Emax]
      if len(E)>1:
        #print('Warning: spline_roots gives multiple roots in given interval')
        pass

      print(new_roots,Emin,Emax)
      print(E)
      E = E[0]
      
      count += 1
      if count==10:
        print('Error in root_finder: Max # of iterations reached. Last window:',[Emin,Emax])
        print('Returning E = ',E)
        break
      print('Root in range',[Emin,Emax],', next guess: E = ',E)
    
    if count<10:
      print('Root found after',count,'iterations: E = ',E)      
    final_roots_list[i] = E
  return final_roots_list



# Root finder for specified interval
def root_finder_secant(E_list, eig_array_list, f_eigs, inputs, E0=0, order=3):
  small_eigs = small_eig_list(eig_array_list)

  eig_fit = np.poly1d(np.polyfit(E_list,small_eigs,order))
  if E0==0:
    E0 = eig_fit.r
    if len(E0)==0 and order == 2:
      E0 = np.polyder(eig_fit).r
    E0 = E0[0]
    print('First guess: E=',E0)

  def smallest_eig(E):
    test_eigs = f_eigs(E,*inputs)
    smallest = test_eigs[0]
    for eig in test_eigs:
      if abs(eig)<abs(smallest):
        smallest = eig
    return abs(smallest)

  
  return secant(smallest_eig,E0,tol=1e-8,maxiter=20)


# Secant root-finding method
def secant(func,x0,tol=1e-8,maxiter=20):
  p0 = x0
  dx = 1e-4
  if x0 >= 0:
    p1 = x0*(1 + dx) + dx
  else:
    p1 = x0*(1 + dx) - dx
  q0 = func(p0)
  q1 = func(p1)
  for iter in range(maxiter):
    if q1 == q0:
      if p1 != p0:
        msg = "Tolerance of %s reached" % (p1 - p0)
      return (p1 + p0)/2.0
    else:
      p = p1 - q1*(p1 - p0)/(q1 - q0)
    if abs(p - p1) < tol:
      return p
    print('Next guess: E=', p)
    p0 = p1
    q0 = q1
    p1 = p
    q1 = func(p1)
  msg = "Failed to converge after %d iterations, value is %s" % (maxiter, p)
  raise RuntimeError(msg)



#################################################################
# Graveyard (old code)
#################################################################
# First two functions not useful since F3 connects different shells & \ell's

# Create list of eigenvalues of F3i in irrep I for single shell & l
def F3i_I_eigs_list_o_l( E_list, L, F3_list, I, shell, l ):
  out = []
  for i in range(len(E_list)):
    E = E_list[i]
    F3 = F3_list[i]
    
    F3_I = proj.irrep_proj_o_l(F3,E,L,I,shell,l) # works for all shells
    if F3_I.shape==():
      F3i_I = 1/F3_I
      F3i_I_eigs = F3i_I
    else:
      F3i_I = defns.chop(LA.inv(F3_I))
      F3i_I_eigs = defns.chop(np.array(sorted(LA.eigvals(F3i_I))).real)

    # F3i = defns.chop(LA.inv(F3)) # temporary BLEH
    # F3i_I = proj.irrep_proj_o(F3i,E,L,I,shell) # temporary BLEH
    # F3i_I_eigs = defns.chop(np.array(sorted(LA.eigvals(F3i_I))).real) # temporary BLEH
      
    out.append( F3i_I_eigs )
  return out

# Create list of eigenvalues of F3i in irrep I for single shell (contains both l=0 and l=2)
def F3i_I_eigs_list_o( E_list, L, F3_list, I, shell ):
  out = []
  for i in range(len(E_list)):
    E = E_list[i]
    F3 = F3_list[i]
    
    F3_I = proj.irrep_proj_o(F3,E,L,I,shell) # works for all shells
    if F3_I.shape==():
      F3i_I = 1/F3_I
      F3i_I_eigs = F3i_I
    else:
      F3i_I = defns.chop(LA.inv(F3_I))
      F3i_I_eigs = defns.chop(np.array(sorted(LA.eigvals(F3i_I))).real)

    # F3i = defns.chop(LA.inv(F3)) # temporary BLEH
    # F3i_I = proj.irrep_proj_o(F3i,E,L,I,shell) # temporary BLEH
    # F3i_I_eigs = defns.chop(np.array(sorted(LA.eigvals(F3i_I))).real) # temporary BLEH
      
    out.append( F3i_I_eigs )
  return out


##################################################
# Old A1+ projection code

# Create list of eigenvalues of F3i_A1 (contains both l=0 and l=2)
# Assumes fixed L w/ 2 shells open
def F3i_A1_eigs_list( E_list, L, F3_list ):
  out = []
  for i in range(len(E_list)):
    E = E_list[i]
    F3 = F3_list[i]
    
    #F3_A1 = proj.A1_proj_old(F3,E,L) # only works for 2 shells
    F3_A1 = proj.A1_proj(F3,E,L) # works for all shells
    if F3_A1.shape==():
      F3i_A1 = 1/F3_A1
      F3i_A1_eigs = F3i_A1
    else:
      F3i_A1 = defns.chop(LA.inv(F3_A1))
      F3i_A1_eigs = defns.chop(np.array(sorted(LA.eigvals(F3i_A1))).real)
      
    out.append( F3i_A1_eigs )
  return out


###################################################
# BAD iso approx code (need to take l'=l=0 part of all matrices inside F3, not just at end)

# Create list of eigenvalues of F3i_iso
# Assumes fixed L w/ 2 shells open
def F3i_iso_eigs_list_bad( E_list, L, F3_list ):
  out = []
  for i in range(len(E_list)):
    E = E_list[i]
    F3 = F3_list[i]
    
    F3_iso = proj.iso_proj_bad(F3,E,L)
    if F3_iso.shape==():
      F3i_iso = 1/F3_iso
      F3i_iso_eigs = F3i_iso
    else:
      F3i_iso = defns.chop(LA.inv(F3_iso))
      F3i_iso_eigs = np.array(sorted(LA.eigvals(F3i_iso))).real

    out.append( F3i_iso_eigs )
  return out
