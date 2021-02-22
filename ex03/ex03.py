import os

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import norm
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve

mat_file_path = os.path.join('resources', 'exercise04_matrices.mat')

mat_file = sio.loadmat(mat_file_path)
M = mat_file['Me']
D = mat_file['De']
K = mat_file['Ke']
B = mat_file['Be']
C = mat_file['Ce']
V = mat_file['V']


def calculateSystemResponse(M, D, K, B, C, s_array, V=None):
    if V is not None:
        M = V.transpose() @ M @ V
        D = V.transpose() @ D @ V
        K = V.transpose() @ K @ V
        B = V.transpose() @ B
        C = C @ V

    H = [None] * len(s_array)
    H_norm = [None] * len(s_array)

    for i, s in enumerate(s_array):
        H_i = C @ spsolve((s ** 2 * M + s * D + K), B)

        if issparse(H_i):
            H_i = H_i.A

        H[i] = H_i
        H_norm[i] = norm(H[i], 'fro')

    return H, H_norm

# a)
f_range = np.linspace(start=2 * np.pi * 1, stop=2 * np.pi * 10000, num=500)
s_array = 1j * f_range

H, H_norm = calculateSystemResponse(M, D, K, B, C, s_array)
H_r, H_r_norm = calculateSystemResponse(M, D, K, B, C, s_array, V)

_, ax = plt.subplots(figsize=(21, 9), nrows=1, ncols=1)
ax.set_title('Frobenius norm of transfer matrix')
ax.semilogy(s_array / 2 / np.pi / 1j, H_norm, 'k', label='original')
ax.semilogy(s_array / 2 / np.pi / 1j, H_r_norm, '--r', label='reduced')
ax.set_xlabel('f [Hz]')
ax.set_ylabel('||H(i 2\pi f)||_{F} [-]')
ax.legend()
plt.show()

error_rel = [None] * len(s_array)
for i in range(len(s_array)):
    error_rel[i] = norm(H[i] - H_r[i]) / norm(H[i], 'fro')

_, ax = plt.subplots(figsize=(21, 9), nrows=1, ncols=1)
ax.set_title('Modal reduction')
ax.semilogy(s_array / 2 / np.pi / 1j, error_rel, 'k')
ax.set_xlabel('f [Hz]')
ax.set_ylabel('\epsilon_F(i 2\pi f) [-]')
plt.show()

# c)
H_E_2 = np.zeros(len(s_array))
H_E_fro = np.zeros(len(s_array))

for i in range(len(s_array)):
    H_E_2[i] = norm(H[i] - H_r[i], 2)
    H_E_fro[i] = norm(H[i] - H_r[i], 'fro')

H2_norm = np.sqrt(1 / 2 / np.pi * 2 * np.trapz(H_E_fro ** 2, x=f_range))
Hinf_norm = max(H_E_2)

print(f'\t||H_E||_H_inf: \t{Hinf_norm}\n\t||H_E||_H_2: \t{H2_norm}')


