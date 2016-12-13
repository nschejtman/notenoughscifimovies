# cython: profile=True
cimport cython
cimport numpy as np
import numpy as np
import scipy.sparse as sps

@cython.boundscheck(False)
def calculate_error_matrix(R, P, Q):
    cdef double [:] err_data = np.zeros(R.data.shape[0], dtype=np.float64), r_data = R.data
    tup = R.nonzero()
    cdef int [:]rows = tup[0], cols = tup[1]
    cdef int i, n_inters = R.data.shape[0]
    for i in range(n_inters):
        err_data[i] = r_data[i] - P[rows[i], :].dot(Q[cols[i], :])
    return sps.csr_matrix((err_data, (rows, cols)), R.shape, dtype=np.float64)