#
#  Code changed from Laurens van der Maaten
#

import numpy as np
import pylab as plt


def Hbeta(D = np.array([]), beta = 1.0):
    """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X = np.array([]), tol = 1e-5, perplexity = 30.0):
    """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point ", i, " of ", n, "...")

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] *= 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] /= 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    return P


def pca(X, no_dims=50):
    """Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

    X = X - np.mean(X, 0)
    (l, M) = np.linalg.eig(np.dot(X.T, X)/X.shape[0])
    Y = np.dot(X, M[:, 0:no_dims])
    l = l.real
    variance = sum(l[:no_dims]) / sum(l)
    return Y.real, variance


class TSNE(object):
    """
    A wrapper class for tsne
    """
    MIN_GAIN = 0.01

    def __init__(self, n_dims, perplexity, lr=50):
        self.n_dims = int(n_dims)
        self.perplexity = perplexity
        self.init_dims = 50
        self.lr = lr
        self.X = None
        self.Y = None
        self.iter = 0
        self.best_error = np.inf
        self.best_sol = None
        self.P = None
        self.iY = None
        self.dY = None
        self.gains = None

    def set_inputs(self, X, init_dims=50):
        """
        Set the inputs data
        :param X: 2D np.ndarray, shaped (instance_num, feature_size)
        :param init_dims: pca X into a smaller dimension for speed
        :return: None
        """
        self.init_dims = init_dims
        self.X = X
        self.run_init()

    def run_init(self):
        # Initialize X
        if self.X.shape[1] != self.init_dims:
            print("doing PCA...")
            self.X, variance = pca(self.X, self.init_dims)
            print('PCA kept {:f}% of variance'.format(variance*100))
        self.Y = np.random.randn(*(self.sol_shape))
        # Compute P-values
        P = x2p(self.X, 1e-5, self.perplexity)
        P += np.transpose(P)
        P /= np.sum(P) * 0.25
        self.P = np.maximum(P, 1e-12)
        self.dY = np.zeros(self.sol_shape)
        self.iY = np.zeros(self.sol_shape)
        self.gains = np.ones(self.sol_shape)

    def run(self, max_iter=1000, record=False):
        Ys = []
        for i in range(max_iter):
            cost = self.step(1)
            if (i+1) % 10 == 0:
                print("iteration {:d}/{:d}, error: {:f}".format(i+1, max_iter, cost))
                if record:
                    Ys.append(self.get_solution())
        return Ys

    def step(self, n=10, lr=None):

        lr = self.lr if lr is None else lr
        cost = None
        for i in range(n):
            # Compute pairwise affinities
            cost, dY = self.cost_gradient(self.Y)
            if cost < self.best_error:
                self.best_error = cost
                self.best_sol = self.Y
            # Perform the update
            self.gains = (self.gains + 0.2) * ((dY > 0) != (self.iY > 0)) + (self.gains * 0.8) * ((dY > 0) == (self.iY > 0))
            self.gains[self.gains < self.MIN_GAIN] = self.MIN_GAIN
            self.iY = self.momentum * self.iY - lr * (self.gains * dY)
            Y = self.Y + self.iY
            self.Y = Y - np.tile(np.mean(Y, 0), (self.n_points, 1))

            self.iter += 1
        return cost

    def cost_gradient(self, Y):
        """
        calculate cost(error) and gradients of given Y
        :param Y: self.Y
        :return: a pair (cost, gradient)
        """
        n = self.n_points
        P = self.P
        if self.iter == 100:
            self.P /= 4
        sum_Y = np.sum(np.square(Y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        # Compute gradient
        PQ = P - Q
        dY = np.zeros(self.sol_shape)
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (self.n_dims, 1)).T * (Y[i, :] - Y), 0)
        cost = np.sum(P * np.log(P / Q))
        return cost, dY

    @property
    def momentum(self):
        return 0.5 if self.iter < 250 else 0.7

    @property
    def sol_shape(self):
        return self.X.shape[0], self.n_dims

    @property
    def n_points(self):
        return self.X.shape[0]

    def get_solution(self):
        return self.Y

    def get_best_solution(self):
        return self.best_sol
