import numpy as np
from ast import literal_eval


data_dir = 'Data/a0=0.1_r0=0_P0=0_a2=1/'

L_list = np.arange(5.0,6.21,0.2)
L_list = [round(L,1) for L in L_list]

A1_root_list = []
iso_root_list = []
diff_list = []
for L in L_list:
  A1_file = data_dir+'A1_roots_L='+str(L)+'.dat'
  iso_file = data_dir+'iso_roots_L='+str(L)+'.dat'

  with open(A1_file,'r') as f_A1:
    A1_root = literal_eval(f_A1.read())[0]
    A1_root_list.append(A1_root)
  
  with open(iso_file,'r') as f_iso:
    iso_root = literal_eval(f_iso.read())[0]
    iso_root_list.append(iso_root)

  diff_list.append(iso_root-A1_root)

