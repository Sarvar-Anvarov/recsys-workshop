import numpy as np
from numpy.linalg import norm

from random import normalvariate
from math import sqrt


# SVD ---
def randomUnitVector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    theNorm = sqrt(sum(x * x for x in unnormalized))
    return [x / theNorm for x in unnormalized]


def svd_1d(A, epsilon=1e-10):
    """ The one-dimensional SVD """

    m, n = A.shape
    x = randomUnitVector(min(m, n))
    lastV = None
    currentV = x

    if m > n:
        B = np.dot(A.T, A)
    else:
        B = np.dot(A, A.T)

    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        currentV_unnormalized = np.dot(B, lastV)
        sigma_squared = norm(currentV_unnormalized)
        currentV = currentV_unnormalized / sigma_squared

        if abs(np.dot(currentV, lastV)) > 1 - epsilon:
            print("converged in {} iterations!".format(iterations))
            return currentV, sigma_squared ** 0.5


def svd(A, k=None, epsilon=1e-10):
    """
        Compute the singular value decomposition of a matrix A
        using the power method. A is the input matrix, and k
        is the number of singular values you wish to compute.
        If k is None, this computes the full-rank decomposition.
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    svdSoFar = []
    if k is None:
        k = min(m, n)

    for i in range(k):
        matrixFor1D = A.copy()

        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D -= singularValue * np.outer(u, v)

        if n < m:
            v, sigma = svd_1d(matrixFor1D, epsilon=epsilon)
            u_unnormalized = np.dot(A, v)
            u = u_unnormalized / sigma

        else:
            u, sigma = svd_1d(matrixFor1D, epsilon=epsilon)
            v_unnormalized = np.dot(A.T, u)
            v = v_unnormalized / sigma

        svdSoFar.append((sigma, u, v))

    singularValues, U, V = [np.array(x) for x in zip(*svdSoFar)]  # the way they stacked need to be transposed
    return singularValues, U.T, V.T


# ALS ---
def nonzeros(m, row):
    """returns the non zeroes of a row in csr_matrix"""
    for index in range(m.indptr[row], m.indptr[row + 1]):
        yield m.indices[index], m.data[index]


def als(A, factors, regularization, n_iterations=50, verbose=False):
    users, items = A.shape

    X = np.random.rand(users, factors) * 0.01
    Y = np.random.rand(items, factors) * 0.01
    current = np.dot(X, Y.T)

    A_T = A.T.tocsr()

    n = 0
    while True:
        least_squares(A, X, Y, regularization)
        least_squares(A_T, Y, X, regularization)

        last = current
        current = np.dot(X, Y.T)

        # calculate errors/might lead to memory error on big dataset as "A" 
        # need to be turned into array from sparce matrix
        if verbose:
            print(
                f"Square error on iteration #{n}:",
                np.sum(np.square((A.toarray() - np.dot(X, Y.T)) * (A.toarray() > 0)))
            )

        if (np.sum(np.abs(last - current)) < 1e-5) or (n >= n_iterations):
            break

        n += 1

    return X, Y


def least_squares(A, X, Y, regularization):
    users, factors = X.shape

    for u in range(users):
        # accumulate YtCuY + regularization * I in A
        C = regularization * np.eye(factors)

        # accumulate YtCuPu in b
        b = np.zeros(factors)

        for i, confidence in nonzeros(A, u):
            factor = Y[i]
            C += np.outer(factor, factor)
            b += confidence * factor

        # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
        X[u] = np.linalg.solve(C, b)  # np.linalg.inv(C).dot(b)
