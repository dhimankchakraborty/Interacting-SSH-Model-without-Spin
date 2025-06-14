'''
There are multiple electrons in the system
This code if for only a single value of "v" and "w"
N : number of unit cells
tot_sites : Total sites in the system
N_e : number of electrons in the system
'''
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import itertools as itr
import h5py
import os
from datetime import datetime
from functions import *


N = 7
w = 1
v = 0.6
tot_sites = 2 * N
topo_state_no = 2
N_e = N

start_time = datetime.now()

topo_state_arr_idx_arr = []
topo_state_arr = []
topo_state_pos_sp_arr = np.zeros((topo_state_no, tot_sites))

basis_set = basis_generator_mul_e(tot_sites, N_e)
hamiltonian = hamiltonian_generator_mul_e(basis_set, tot_sites, v, w)

dim = len(hamiltonian)

e_val_arr, e_vec_arr = np.linalg.eigh(hamiltonian)

e_val_arr_abs = np.abs(e_val_arr)

e_val_arr_abs_sorted = np.sort(e_val_arr_abs)

for i in range(topo_state_no):
    topo_state_arr_idx_arr.append((e_val_arr_abs == e_val_arr_abs_sorted[i]).argmax())

sites_arr = np.arange(tot_sites)

for i in topo_state_arr_idx_arr:
    topo_state_arr.append(e_vec_arr[:, i])
topo_state_arr = np.array(topo_state_arr)

for i, state in enumerate(topo_state_arr):
    for j, element in enumerate(state):
        topo_state_pos_sp_arr[i] += element * basis_set[j]
    topo_state_pos_sp_arr[i] = normalize(topo_state_pos_sp_arr[i])

end_time = datetime.now()
time_taken = end_time - start_time
print(f"Time taken: {time_taken}")

for i, wf in enumerate(topo_state_pos_sp_arr):
    plt.plot(sites_arr, wf, label=f'{i}----{topo_state_arr_idx_arr[i]}')
plt.legend()
plt.grid()
plt.show()

for i, wf in enumerate(topo_state_pos_sp_arr):
    plt.plot(sites_arr, wf**2, label=f'{i}----{topo_state_arr_idx_arr[i]}')
plt.legend()
plt.grid()
plt.show()


data_file_name = "ssh-mul-e-1-run.h5"
data_file_output_directory = "data"
group_name = f"{N}-{v}-{w}-{topo_state_no}-{datetime.now().strftime("%d-%m-%y-%H-%M-%S")}"

os.makedirs(data_file_output_directory, exist_ok=True)
data_file_full_path = os.path.join(data_file_output_directory, data_file_name)

file = h5py.File(data_file_full_path, 'a')

group = file.create_group(group_name)

group.attrs['Number of Unit Cells'] = N
group.attrs['Total number of Sites'] = tot_sites
group.attrs['Number of electrons in the system'] = N_e
group.attrs['v'] = v
group.attrs['w'] = w
group.attrs['No of Recorded Topological States'] = topo_state_no
group.attrs['Dimension of Hilbert Space'] = dim
group.attrs['Date'] = datetime.now().strftime("%d-%m-%y")
group.attrs['Time'] = datetime.now().strftime("%H:%M:%S")
group.attrs['Time Taken'] = str(time_taken)

e_val_dset = group.create_dataset("e_val_arr", data=e_val_arr)
e_val_dset.attrs['Shape'] = np.shape(e_val_arr)
e_val_dset.attrs['Style'] = "I1" 
e_val_dset.attrs['I1'] = "Denotes eigen value of each eigenstate" 

topo_state_dset = group.create_dataset("topo_state_pos_sp_arr", data=np.array(topo_state_pos_sp_arr))
topo_state_dset.attrs['Shape'] = np.shape(topo_state_pos_sp_arr)
topo_state_dset.attrs['Style'] = "I1xI2" 
topo_state_dset.attrs['I1'] = "Denotes each zero energy eigenstate" 
topo_state_dset.attrs['I2'] = "Denotes each element of a ssh_1_e_mul_w_run_v1.pyzero energy eigenstate" 

file.close()

