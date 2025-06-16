import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from itertools import combinations


def basis_set_generator(tot_sites, N_particles): # Checked OK
    basis = []

    for comb in combinations(range(tot_sites), N_particles):
        state = [0] * tot_sites
        for idx in comb:
            state[idx] = 1
        basis.append(state)
    
    return np.array(basis)


def state_idx_mapping(basis_set): # Checked OK
    return {tuple(state) : i for i, state in enumerate(basis_set)}


def creation_operator(state, i): # Checked OK
    if state[i] == 1:
        return 0, None
    
    res_state = state.copy()
    res_state[i] = 1

    sign = 1
    for j in range(i):
        if res_state[j] == 1:
            sign *= -1
    
    return sign, res_state


def annihilation_operator(state, i): # Checked OK
    if state[i] == 0:
        return 0, None
    
    res_state = state.copy()
    res_state[i] = 0

    sign = 1
    for j in range(i):
        if res_state[j] == 1:
            sign *= -1
    
    return sign, res_state


def hopping_operator(state, initial_position, target_position): # Checked OK
    sign_1, state_1 = annihilation_operator(state, initial_position)

    # if sign_1 == 0:
    #     return 0, None

    sign_2, state_2 = creation_operator(state_1, target_position)

    return sign_1 * sign_2, state_2


def hamiltonian_matrix_generator(basis_set, tot_sites, v, w, V1, V2): # Checked OK
    dim = len(basis_set)
    hamiltonian = np.zeros((dim, dim), dtype=np.float64)

    state_idx_dict = state_idx_mapping(basis_set)

    for i, state in enumerate(basis_set):
        for j in range(tot_sites):
            if state[j] == 0:
                continue

            if j + 1 < tot_sites:
                if state[j + 1] == 0:
                    initial_position = j
                    target_position = j + 1
                    # print(initial_position, target_position)
                    sign, res_state = hopping_operator(state, initial_position, target_position)
                    res_state_idx = state_idx_dict[tuple(res_state)]

                    if j % 2 == 0:
                        hamiltonian[res_state_idx, i] += v * sign
                    else:
                        hamiltonian[res_state_idx, i] += w * sign
                
                elif state[j + 1] == 1:
                    if j % 2 == 0:
                        hamiltonian[i, i] += V1
                    else:
                        hamiltonian[i, i] += V2

            if j - 1 >= 0:
                if state[j - 1] == 0:
                    initial_position = j
                    target_position = j - 1
                    sign, res_state = hopping_operator(state, initial_position, target_position)
                    res_state_idx = state_idx_dict[tuple(res_state)]

                    if j % 2 == 0:
                        hamiltonian[res_state_idx, i] += w * sign
                    else:
                        hamiltonian[res_state_idx, i] += v * sign
                
                elif state[j - 1] == 1:
                    if j % 2 == 0:
                        hamiltonian[i, i] += V2
                    else:
                        hamiltonian[i, i] += V1
    
    return hamiltonian


def normalize(vec):
  norm = np.linalg.norm(vec)
  if norm == 0:
    return vec
  return vec / norm
