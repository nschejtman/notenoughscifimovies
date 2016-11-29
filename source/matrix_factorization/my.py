import numpy as np
import sys
sys.path.append('./../')
import utils.utils as utils
import time
import scipy.sparse as sps


def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02, iteration_size=2500):
    Q = Q.T
    for step in xrange(steps):
        E = sps.csr_matrix(R, copy=True, dtype=P.dtype)
        tup = E.nonzero()
        rows, cols = tup[0], tup[1]
        # st = time.time()
        for i in range(E.data.shape[0]):
            E[rows[i],cols[i]] -= P[rows[i],:].dot(Q[:,cols[i]])
        # et = time.time()
        # print et-st

        # Construct sparse error matrix in batches
        '''st = time.time()
        E = None
        n_iterations = R.shape[0] / iteration_size + (R.shape[0] % iteration_size != 0)
        for i in range(n_iterations):
            print "Iteration: ", i + 1, "/", n_iterations
            start = i * iteration_size
            end = start + iteration_size if i < n_iterations - 1 else R.shape[0]

            P_partition = P[start:end,:]

            R_partition = R[start:end,:]
            E_partition = np.array(R_partition - P_partition.dot(Q))

            E_partition[(R_partition != 0).toarray() == False] = 0.0
            E_partition = sps.csr_matrix(E_partition)

            if E is None:
                E = E_partition.copy()
            else:
                E = sps.vstack([E, E_partition])

        et = time.time()
        print et-st'''

        # Updated P, Q
        P_prime = (1 - alpha * beta) * P + alpha * E.dot(Q.T)
        Q_prime = (1 - alpha * beta) * Q + alpha * E.T.dot(P).T

        P, Q = P_prime.copy(), Q_prime.copy()
        del Q_prime, P_prime

        # Calculate error
        print "Calculate error"
        error = np.sum(E.data ** 2)/2 + beta*np.sum(P ** 2)/2 + beta*np.sum(Q ** 2)/2

        print error

    return P, Q.T


def main():
    np.random.seed(1)
    K = 20
    R = utils.read_interactions()[:5000]
    P = np.random.rand(R.shape[0], K)
    Q = np.random.rand(R.shape[1], K)
    matrix_factorization(R, P, Q, K)


main()
