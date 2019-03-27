import numpy as np
sqrt=np.sqrt; norm=np.linalg.norm
import defns

# First O(Delta^3) term in K3
def K3cubicA(E,pvec,lp,mp,kvec,l,m):
  p=norm(pvec); k=norm(kvec)
  Ep = defns.E2k(E,p); Ek=defns.E2k(E,k)
  wp = defns.omega(p); wk = defns.omega(k)
  ap = defns.qst(E,p); a = defns.qst(E,k)
  pterm = E*wp-3; kterm = E*wk-3

  if lp==l==mp==m==0:
    D3p = Ep**2-4; D3 = Ek**2-4
    out = D3p**3 + D3**3 + 2*(pterm**3+kterm**3) + 8*E**2*( pterm*(p*ap/Ep)**2 + kterm*(k*a/Ek)**2 )
  elif lp==0 and l==2:
    #out = 16/5 * kterm * (E*a/Ek)**2 * defns.y2real(kvec,m)
    out = 16/5 * kterm * (E/Ek)**2 * defns.y2real(kvec,m)  # removed a=qk* factor (no q)
  elif lp==2 and l==0:
    #out = 16/5 * pterm * (E*ap/Ep)**2 * defns.y2real(pvec,m)
    out = 16/5 * pterm * (E/Ep)**2 * defns.y2real(pvec,mp) # removed ap=qp* factor (no q)
  else:
    out = 0

    #out *= ap**lp * a**l   # q factors are NOT included here (no q)

  if out.imag>1e-15:
    print('Error: imaginary part in K3cubicA')
  return out.real
