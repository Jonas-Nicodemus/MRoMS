import time

from scipy.integrate import solve_ivp, ode
from scipy.io.harwell_boeing import hb_read, hb_write
from scipy import sparse, linalg
from scipy.io import loadmat
from pymatreader import read_mat
import matplotlib.pyplot as plt
import numpy as np
import re
import os

from scipy.linalg import solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, inv


def conv(value):
    try:
        return float(value)
    except ValueError:
        return float(re.sub(r"D", 'E', str(value, 'utf-8')))


def read_hb(file_path):
    with open(file_path) as file:
        line = file.readline().strip("\n")
        if not len(line) > 72:
            raise ValueError("Expected at least 72 characters for first line, "
                             "got: \n%s" % line)
        title = line[:72]
        key = line[72:]

        # Second line
        line = file.readline().strip("\n")
        if not len(line.rstrip()) >= 56:
            raise ValueError("Expected at least 56 characters for second line, "
                             "got: \n%s" % line)
        total_nlines = int(line[:14])
        pointer_nlines = int(line[14:28])
        indices_nlines = int(line[28:42])
        values_nlines = int(line[42:56])

        data = np.loadtxt(file_path, skiprows=5, converters={0: conv})

        indptr = data[:pointer_nlines].astype(int) - 1
        indices = data[pointer_nlines:pointer_nlines + indices_nlines].astype(int) - 1
        values = data[pointer_nlines + indices_nlines:pointer_nlines + indices_nlines + values_nlines]

        sparse_matrix = csc_matrix((values, indices, indptr))

        # Symmetrization of scipy sparse matrices
        rows, cols = sparse_matrix.nonzero()
        sparse_matrix[cols, rows] = sparse_matrix[rows, cols]

    return sparse_matrix


def read_hb_from_mat(file_path, name):
    mat_dic = read_mat(file_path)[name]
    indptr = mat_dic['jc']
    indices = mat_dic['ir']
    values = mat_dic['data']
    return csc_matrix((values, indices, indptr))


def apply_dirichlet_conditions(A, indices):
    A_dense = A.A
    A_dense = np.delete(A_dense, indices, axis=0)
    A_dense = np.delete(A_dense, indices, axis=1)

    return csc_matrix(A_dense)


def f_factory(f0, frequency=1, amplitude=0.01):
    force_fcn = lambda t: f0 * amplitude * np.sin(2 * np.pi * frequency * t)
    return force_fcn


def ode_f(t, x, M, K, f_fcn):
    u, dudt = x.reshape((2, -1))

    dxdt = np.concatenate((dudt, spsolve(M, f_fcn(t).A[:, 0] - K @ u)))

    return dxdt


# def ode_jac(t, x, M_inv, K):
#     order = K.shape[0]
#     upper = (np.zeros((order, order)), np.ones((order, order)))
#     lower = (-M_inv @ K, np.zeros((order, order)))
#
#     return np.vstack((np.hstack(upper), np.hstack(lower)))


resources_path = os.path.join('ex02', 'resources')

K_L_path = os.path.join(resources_path, 'K_L.txt')
M_L_path = os.path.join(resources_path, 'M_L.txt')

K_txt = read_hb(K_L_path)

K_mat_path = os.path.join(resources_path, 'K.mat')
M_mat_path = os.path.join(resources_path, 'M.mat')

K_mat = read_hb_from_mat(K_mat_path, 'K')
M_mat = read_hb_from_mat(M_mat_path, 'M')

is_equal = np.allclose(K_txt.A, K_mat.A)

# print(f'{K_txt} \n {K_mat}')
#
# fig, axes = plt.subplots(1, 2)
# ax = axes.ravel()
# ax[0].spy(K_txt)
# ax[0].set_title("Sparsity of K_txt")
# ax[1].spy(K_mat)
# ax[1].set_title("Sparsity of K_mat")
# plt.show()

### Apply dirchlet
# Dirichlet conditions in all directions at nodes 27,28,29,48,49,50.
# node    27 28 29 47 48 49 50 51
# x       502 505 535 64 532 499 496 4
# y       503 506 536 65 533 500 497 5
# z       504 507 537 66 534 501 498 6

K = K_mat
M = M_mat

dirichlet_indices = np.array(
    [502, 503, 504, 505, 506, 507, 535, 536, 537, 532, 533, 534, 499, 500, 501, 496, 497, 498]) - 1

K = apply_dirichlet_conditions(K, dirichlet_indices)
M = apply_dirichlet_conditions(M, dirichlet_indices)

order = K.shape[0]

#####

force_indices = np.array([65, 5]) - 1

f = csc_matrix((order, 1))
f[force_indices] = 100

u_sparse = spsolve(K, f)

##### e)
time_domain = [0, 2]
x0 = np.zeros(2 * order)

force_indices = np.array([65, 5]) - 1
f0 = csc_matrix((order, 1))
f0[force_indices] = 1

f_fcn = f_factory(f0)

# M_inv = inv(M)  # scipy.sparse.linalg.inv

# jac = csc_matrix((2 * order, 2 * order))
# jac[:order, order:] = np.ones((order, order))
# jac[order:, :order] = -M_inv @ K

start_time = time.time()
ivp_sol = solve_ivp(ode_f, (0, 0.001), x0, method='RK23', args=(M, K, f_fcn))
end_time = time.time() - start_time

print('END')
