import math
import time
import numpy as np
from numba import njit


@njit(fastmath=True)
def f(Lambda,eta):
    res = 0.
    eta2=eta**2
    for x in range(-Lambda,Lambda+1):
        for y in range(-Lambda,Lambda+1):
            for z in range(-Lambda,Lambda+1):
                if(x**2+y**2+z**2 < Lambda**2):
                    res += 1/(x**2+y**2+z**2-eta2)

    return res - 4*math.pi*Lambda

#for Lambda in range(2000,2100,100):
#    start = time.time()
#    print(Lambda,f(Lambda,0.121))
#    end = time.time()
#    print('time is:', end - start, ' s')


data = [[30.0, 2.7779870297280973],[35.0, 2.7649857172188868],[40.0, 2.7567306788796118],[70.0, 2.732290757902254]]
E = 2.7779870297280973
L = 30.0
a0 = 2.
MB = 2*np.sqrt(1 - 1/a0**2)


for L,E in data:
    p = np.sqrt((1 - 2*E**2 + E**4 - 2*MB**2 -2*E**2*MB**2 + MB**4 )/(4*E**2))
    eta = p*L/2/math.pi
    #print(eta)
    kcotd = (1./math.pi/L)*f(2200,eta)
    print(p,kcotd,L)


