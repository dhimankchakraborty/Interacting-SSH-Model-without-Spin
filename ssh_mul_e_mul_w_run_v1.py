'''
There are multiple electrons in the system
This code if for only a single value of "v" and multiple values of "w"
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


w_step_no = 101
w_0 = 0
w_end = 3
w_arr = np.linspace(w_0, w_end, w_step_no)
w_arr = np.around(w_arr, decimals=3)
v = 1
N = 4
tot_sites = 2 * N
N_e = N
topo_state_no = 2

start_time = datetime.now()



all_e_val_arr = []
topo_state_pos_sp_arr = np.zeros((w_step_no, topo_state_no, tot_sites))

for k, w in enumerate(w_arr):
    topo_state_arr = []
    topo_state_arr_idx_arr = []

    basis_set = basis_generator_mul_e(tot_sites, N_e)
    hamiltonian = hamiltonian_generator_mul_e(basis_set, tot_sites, v, w)

    dim = len(hamiltonian)

    e_val_arr, e_vec_arr = np.linalg.eigh(hamiltonian)

    all_e_val_arr.append(e_val_arr)

    e_val_arr_abs = np.abs(e_val_arr)

    e_val_arr_abs_sorted = np.sort(e_val_arr_abs)

    for i in range(topo_state_no):
        topo_state_arr_idx_arr.append((e_val_arr_abs == e_val_arr_abs_sorted[i]).argmax())

    for i in topo_state_arr_idx_arr:
        topo_state_arr.append(e_vec_arr[:, i])
    
    for i, state in enumerate(topo_state_arr):
        for j, element in enumerate(state):
            topo_state_pos_sp_arr[k][i] += element * basis_set[j]
        topo_state_pos_sp_arr[k][i] = normalize(topo_state_pos_sp_arr[k][i])

all_e_val_arr = np.array(all_e_val_arr)

end_time = datetime.now()
time_taken = end_time - start_time
print(f"Time taken: {time_taken}")


for i, e_w_arr in enumerate(all_e_val_arr.transpose()):
    plt.plot(w_arr, e_w_arr, label=f"{i}")
plt.legend()
plt.grid()
plt.show()


data_file_name = "ssh-mul-e-mul-w-run.h5"
data_file_output_directory = "data"
group_name = f"{N}-{w_0}-{w_end}-{w_step_no}-{v}-{topo_state_no}-{datetime.now().strftime("%d-%m-%y-%H-%M-%S")}"

os.makedirs(data_file_output_directory, exist_ok=True)
data_file_full_path = os.path.join(data_file_output_directory, data_file_name)

file = h5py.File(data_file_full_path, 'a')

group = file.create_group(group_name)

group.attrs['Number of Unit Cells'] = N
group.attrs['Total number of Sites'] = tot_sites
group.attrs['Number of electrons in the system'] = N_e
group.attrs['Start of w'] = w_0
group.attrs['End of w'] = w_end
group.attrs['Number of w steps'] = w_step_no
group.attrs['v'] = v
group.attrs['No of Recorded Topological States'] = topo_state_no
group.attrs['Dimension of Hilbert Space'] = dim
group.attrs['Date'] = datetime.now().strftime("%d-%m-%y")
group.attrs['Time'] = datetime.now().strftime("%H:%M:%S")
group.attrs['Time Taken'] = str(time_taken)

v_arr_dset = group.create_dataset("w_arr", data=w_arr)
v_arr_dset.attrs['Shape'] = np.shape(w_arr)
v_arr_dset.attrs['Style'] = "I1"
v_arr_dset.attrs['I1'] = "Denotes the value of each w"

e_val_dset = group.create_dataset("e_val_arr", data=all_e_val_arr)
e_val_dset.attrs['Shape'] = np.shape(all_e_val_arr)
e_val_dset.attrs['Style'] = "I1xI2"
e_val_dset.attrs['I1'] = "Denotes the set of eigen values of each w" 
e_val_dset.attrs['I2'] = "Denotes eigen value of each eigenstate" 

topo_state_dset = group.create_dataset("topo_state_pos_sp_arr", data=np.array(topo_state_pos_sp_arr))
topo_state_dset.attrs['Shape'] = np.shape(topo_state_pos_sp_arr)
topo_state_dset.attrs['Style'] = "I1xI2xI3" 
topo_state_dset.attrs['I1'] = "Denotes the set of eigen states of each w" 
topo_state_dset.attrs['I2'] = "Denotes each zero energy eigenstate" 
topo_state_dset.attrs['I3'] = "Denotes each element of a zero energy eigenstate" 

file.close()