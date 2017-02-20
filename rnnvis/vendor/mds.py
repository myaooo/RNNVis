import os, sys, getopt, pdb
import numpy as np
from numpy.linalg import *
from numpy.random import *
import pylab


def mds(d, dimensions=2):
    """
    Multidimensional Scaling - Given a matrix of interpoint distances,
    find a set of low dimensional points that have similar interpoint
    distances.
    """

    E = (-0.5 * d**2)

    # Use mat to get column and row means to act as column and row means.
    Er = np.mat(np.mean(E, 1))
    Es = np.mat(np.mean(E, 0))

    # From Principles of Multivariate Analysis: A User's Perspective (page 107).
    F = np.array(E - np.transpose(Er) - Es + np.mean(E))

    U, S, V = svd(F)

    Y = U * np.sqrt(S)

    return Y[:, 0:dimensions], S


def norm(vec):
    return np.sqrt(sum(vec**2))


def square_points(size):
    nsensors = size ** 2
    return np.array([(i / size, i % size) for i in range(nsensors)])


def test():

    points = square_points(10)

    distance = np.zeros((100,100))
    for (i, pointi) in enumerate(points):
        for (j, pointj) in enumerate(points):
            distance[i, j] = norm(pointi - pointj)

    Y, eigs = mds(distance)

    pylab.figure(1)
    pylab.plot(Y[:, 0], Y[:, 1], '.')

    pylab.figure(2)
    pylab.plot(points[:, 0], points[:, 1], '.')

    pylab.show()


def main():

    def usage():
        print(sys.argv[0] + "[-h] [-d]")

    try:
        (options, args) = getopt.getopt(sys.argv[1:], 'dh', ['help', 'debug'])
    except getopt.GetoptError:
        # print help information and exit:
        usage()
        sys.exit(2)

    for o, a in options:
        if o in ('-h', '--help'):
            usage()
            sys.exit()
        elif o in ('-d', '--debug'):
            pdb.set_trace()

    test()

if __name__ == "__main__":
    main()
