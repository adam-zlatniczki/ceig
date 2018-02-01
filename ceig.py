# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 16:43:50 2018

@author: Adam Zlatniczki (Budapest University of Technology and Economics,
                          Department of Computer Science and Information Theory)
"""

from numpy import dot, concatenate
from scipy.linalg import eig, qr, solve
from scipy.optimize import brentq


def _transform_problem(A, N, t):
    """Transforms the original problem."""
    m = N.shape[1]
    #  n = N.shape[0] - m

    P, R_prime = qr(N)
    R = R_prime[:m, :]

    A_prime = dot(P.T, dot(A, P))
    # B = A_prime[:m, :m]
    G = A_prime[m:, :m]
    C = A_prime[m:, m:]

    y = solve(R, t)

    s2 = 1 - dot(y, y)

    b = -dot(G, y)

    delta, Q = eig(C)
    #D = diag(delta)
    Q_T = Q.T

    d = dot(Q_T, b)

    y = y.reshape((y.shape[0]))
    d = d.reshape((d.shape[0]))
    s2 = s2[0][0]
    delta = delta.reshape((delta.shape[0]))

    return y, s2, delta, Q, d


def f(l, d, delta, s2, k=0):
    """ The secular function. """
    return sum( (d[k:] / (delta[k:] - l))**2 ) - s2


def ceig(A, N, t, min_inf=-1000):
    """
    Solves the constrained eigenvalue problem

        min x^T A x
        st. N^T x = t
            x^T x = 1

    Reference:
        - Gander, Golub & Matt (1989): A Constrained Eigenvalue Problem

    """
    y, s2, delta, Q, d = _transform_problem(A, N, t)

    l = brentq(f, min_inf, min(delta).real * (1.0 - 1e-16), args=(d, delta, s2))

    u = d.reshape((d.shape[0])) / (delta - l)
    z = dot(Q, u)
    x = concatenate((y, z), axis=0)

    return x
