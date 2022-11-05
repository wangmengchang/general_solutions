# python3

"""
General Solutions
to Systems of Linear Equations

Author: Wang Mengchang
wangmengchang@gmail.com

USAGE:

GSS is used in solving nonhomogeneous system of linear equations like Ax = b.

In linear algebra, Ax = b may have multiple solutions when Ax=0 has nontrivial solutions.

Once we find a special solution to Ax = b, and bases of the null space of A (i.e., all solutions to Ax = 0),
any solution to Ax = b can be writen as the special solution plus a linear combination of of the bases.


>>> import numpy as np
>>> x,B,n = gsolve(A, b)                   #to get special solution x, bases (column vectors of B), dimension of null space
>>> if n >= 1:                             #when n<1, Ax = b has the solution x only
>>>     v = np.random.rand(n)              #a vector 
>>>     x1 = x + B.dot(v)                  #a new solution to Ax = b
>>>     print(np.allclose(A.dot(x1), b))   #check whether x1 satisfy A x1 = b
True

"""


import numpy as np
from numpy.random import default_rng

def get_null_space(A, rcond=None):
    """
    Compute the bases of the nullspace of A, i.e., solution space of Ax = 0
    return the nullspace basis matrix Q:=[b1,...,bk]
    each column is a basis, i.e., any affine combination of columns 
        x = sum a_j * b_j
    is a solution to Ax = 0
    """
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q

def gsolve(A, b):
    """
    solve(A, b)
    A: numpy 2-d matrix
    b: numpy array

    return x_p, x_0, I_free
    x_p: a special solution for Ax = b, numpy array or none
    B: basis matrix of the solution space of Ax = 0 , matrix of dim x nullity
    nullity: dimensionality of the nullspace, a random vector of the dimensionality with the nullspace basis matrix
    """
    x_p = (np.linalg.lstsq(A, b, rcond=-1)[0]).flatten()
    B = get_null_space(A)
    _, nullity = B.shape
    if nullity < 1:
        return x_p, None, 0
    else:
        return x_p, B, nullity


def _make_solution_with(x, B, v):
    if B is None:
        return x
    else:
        assert(B.shape[1] == len(v))
        return x + B.dot(v)

def random_solution(A,b):
    x,B,n = gsolve(A, b)
    if B is None or n<1:
        return x
    else:
        v = np.random.rand(n)
        return _make_solution_with(x, B, v)

def gss_usage():
    print("GSS (general solutions) to nonhomogeneous linear equations, Ax=b\n")
    print("Note: any solution to Ax=b can be expressed as the sum of a special solution to Ax=b, plus a solution to Ax=0.\n")
    print("Given A and b, we can get a solution in this way:\n")
    print(">>> import numpy as np")
    print(">>> from general_solutions import *")
    print(">>> x,B,n = gsolve(A, b)")
    print(">>> if n > 0:") 
    print(">>>     v = np.random.rand(n)") 
    print(">>>     x1 = x + B.dot(v)\n")
    print("x1 is a solution to Ax=b, too.\n")
    print(":::: x,B,n = gsolve(A, b) ::::")
    print("    x: is a special solution to Ax=b, obtained by the `numpy.linalg.lstsq()` solver.")
    print("    N: is a matrix of null space bases (as column vectors)")
    print("    n: is the nullity (dimension of the null space of A)")
    print("       NOTE: if n==0, then Ax=b has the special solution x only, no more solutions.") 
    print(":::: x1 = x + B.dot(v) ::::")
    print("    Here v is a random vector, and B.dot(v) is a linear combination of bases.") 
    print("    Then, x1 is another solution to Ax=b.\n")

if __name__ == "__main__":
    A=np.array([[0.28502683, 0.25979706, 0.90550497, 0.48318177, 0.22723662],
       [0.1487263 , 0.38205604, 0.91557374, 0.1736457 , 0.15280735],
       [0.48657981, 0.15292602, 0.65385788, 0.61087277, 0.45238306]])
    b = np.array([2,1,3])
    print(np.allclose(A.dot(random_solution_of(A, b)), b))
