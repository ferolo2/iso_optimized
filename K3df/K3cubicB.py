import numpy as np
sqrt=np.sqrt; norm=np.linalg.norm
import defns; import K3_defns
y2real = defns.y2real; Sspher=K3_defns.Sspher; Uspher=K3_defns.Uspher

# First O(Delta^3) term in K3
def K3cubicB(E,pvec,lp,mp,kvec,l,m):
  #pvec=np.array(pvec); kvec=np.array(kvec)
  p=norm(pvec); k=norm(kvec)
  phat = np.array([x/p for x in pvec]) if p!=0 else pvec
  khat = np.array([x/k for x in kvec]) if k!=0 else kvec

  Ep = defns.E2k(E,p); Ek=defns.E2k(E,k)
  wp = defns.omega(p); wk = defns.omega(k)
  ap = defns.qst(E,p); a = defns.qst(E,k)
  # Bp = p/(E-wp); Bk = k/(E-wk)
  # gp = 1/sqrt(1-Bp**2); gk = 1/sqrt(1-Bk**2)
  k_p = np.dot(pvec,kvec)

  # Denote primed (outgoing) momenta with t
  # Denote p_{12}^+ by p12, and p_{12}^- by m12; add t for primed versions


  pstark_vec = K3_defns.pstark(E,pvec,kvec) #gk*Bk*wp*khat + (gk-1)*p*khat_phat*khat + pvec
  kstarp_vec = K3_defns.pstark(E,kvec,pvec) #gp*Bp*wk*phat + (gp-1)*k*khat_phat*phat + kvec
  p12t_stark_vec = K3_defns.p12stark(E,pvec,kvec) #gk*Bk*(E-wp)*khat - (gk-1)*p*khat_phat*khat - pvec
  p12_starp_vec = K3_defns.p12stark(E,kvec,pvec) #gp*Bp*(E-wk)*phat - (gp-1)*k*khat_phat*phat - kvec

  pstark = norm(pstark_vec);        kstarp = norm(kstarp_vec)
  p12t_stark = norm(p12t_stark_vec); p12_starp = norm(p12_starp_vec)

  p3_p3t = wk*wp - k_p
  p12_p12t = (E-wk)*(E-wp) - k_p
  p12_p3t = (E-wk)*wp + k_p
  p3_p12t = (E-wp)*wk + k_p

  t = K3_defns.T(E,pvec,kvec)
  u = np.outer(p12t_stark_vec, p12_starp_vec)  # tensor needed for full U

  
  if lp==l==mp==m==0:
    out = p3_p3t**3 + 1/4*( (p12_p3t)**3 + (p3_p12t)**3 ) + p12_p3t*pstark**2*a**2 + p3_p12t*kstarp**2*ap**2 \
    + 1/16*p12_p12t**3 + 1/4*p12_p12t*( p12t_stark**2*a**2 + p12_starp**2*ap**2 ) \
    + 1/3*p12_p12t*a**2*ap**2*Sspher(t,0,0,0,0) + 2/3*a**2*ap**2*Uspher(t,u,0,0,0,0)

  elif lp==mp==0 and l==2:
    out = 2/5*p12_p3t*y2real(pstark_vec,m) + 1/10*p12_p12t*y2real(p12t_stark_vec,m) \
    + p12_p12t*ap**2*Sspher(t,0,0,2,m) + 2*ap**2*Uspher(t,u,0,0,2,m) # removed a^2=qk*^2 factor (no q)
    # factor of 1/sqrt(15) that Steve has is inside Sspher and Uspher

  elif lp==2 and l==m==0:
    out = 2/5*p3_p12t*y2real(kstarp_vec,mp) + 1/10*p12_p12t*y2real(p12_starp_vec,mp) \
    + p12_p12t*a**2*Sspher(t,2,mp,0,0) + 2*a**2*Uspher(t,u,2,mp,0,0) # removed ap^2=qp*^2 factor (no q)
    # factor of 1/sqrt(15) that Steve has is inside Sspher and Uspher

  elif lp==l==2:
    out = 3*p12_p12t*Sspher(t,2,mp,2,m) + 6*Uspher(t,u,2,mp,2,m) # removed (ap*a)^2=(qk*qp*)^2 factor (no q)
    # factor of 1/15 that Steve has is inside Sspher and Uspher
  else:
    out = 0

    #out *= ap**lp * a**l   # q factors are NOT included here (no q)

  if out.imag>1e-15:
    print('Error: imaginary part in K3cubicB')
  return out.real
