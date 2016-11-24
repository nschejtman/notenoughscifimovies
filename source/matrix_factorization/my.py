import numpy
import utils.utils as utils
import time
import scipy.sparse as sps


def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02, iteration_size=10000):
    Q = Q.T
    for step in xrange(steps):

        # Construct sparse error matrix in batches
        E = None
        n_iterations = R.shape[0] / iteration_size + (R.shape[0] % iteration_size != 0)
        for i in range(n_iterations):
            print "Iteration: ", i + 1, "/", n_iterations
            start = i * iteration_size
            end = start + iteration_size if i < n_iterations - 1 else R.shape[0]

            P_partition = P[start: end, :]

            R_partition = R[start:end, ]
            E_partition = (R_partition - P_partition.dot(Q)).toarray()

            E_partition[(R_partition != 0).toarray() == False] = 0.0
            E_partition = sps.csr_matrix(E_partition)

            if E is None:
                E = E_partition
            else:
                E = sps.vstack([E, E_partition])

        # Updated P, Q
        P_prime = (1 - alpha * beta) * P + alpha * E.dot(Q.T)
        Q_prime = (1 - alpha * beta) * Q + alpha * E.T.dot(P)

        P, Q = P_prime, Q_prime

        # Calculate error
        error = numpy.sum(E.data ** 2) + numpy.sum(P ** 2) + numpy.sum(Q ** 2)

        print error

    return P, Q.T


def main():
    K = 20
    R = utils.read_interactions()
    P = numpy.random.rand(R.shape[0], K)
    Q = numpy.random.rand(R.shape[1], K)
    matrix_factorization(R, P, Q, K)


main()
