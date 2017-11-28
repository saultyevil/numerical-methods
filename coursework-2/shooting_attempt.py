"""
Usage of the shooting method to solve the boundary value problem:
    y'' + y' + 1 = 0, with boundary condtions:
        y(0) = 0; y(1) = 1.

The BVP has the vector solution,
    q = (y1, y2) = (y, y')
and IVP conditions,
    q(0) = (y(0), y'(0)).
Here, y'(0) is not know, so we set,
    q(0) = (0, z),
where z is an educated guess.

As y1 = y and y2 = y', then,
    y1' = y' = y2, and,
    y2' = y'' = -1 - y' = -1 - y2.
Thus we can write the vector,
    q' = (y2, -1 - y2).

We want to find the residual of the IVP integrated using the initial guess at
the boundary conditions defined as,
    phi(z) = y(1, z) - 1 = y1(1, z) - 1.
When phi(z) = 0, we have found the value of z and thus we can use this value of
z in the vector q(0) to integrate the IVP over a suitable grid to recieve the
solution which we want.
"""

import numpy as np
import scipy as scp
from matplotlib import pyplot as plt


def IVP(q, x):
    """
    Define the IVP q'(x). This is the IVP which will be integrated to find
    the solution y(x).
    """

    dqdx = np.zeros_like(q)
    dqdx[0] = q[1]
    dqdx[1] = -1 - q[1]
    return dqdx


def root_find_func(z):
    """
    Defines the function phi(z) which takes in the intial guess of z and
    creates a residual function. Using rooting finding methods on this function
    will return an appropriate value of z to use in the intial guess vector
    when integrating the IVP to find y(x).
    """

    ivp_int = scp.integrate.odeint(IVP, [0, z], [0, 1])
    bound = ivp_int[-1, 0]
    return bound - 1


def shooting():
    """
    The shooting algorithm. Uses an initial guess z to integrate the IVP and
    then uses a rootfinding alogirthm to find the value of z. A grid is then
    generated and the IVP is integrated again using the guess [0, z_root]
    where z_root is the root of the function phi(z) = y1(1, z) - 1.
    """

    z_guess = 1
    z_root = scp.optimize.root(root_find_func, z_guess).x

    x = np.linspace(0, 1)
    bvp_sol = scp.integrate.odeint(IVP, [0, z_root], x)

    return x, bvp_sol[:, 0]


x, sol = shooting()
plt.plot(x, sol)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.xlim(0, 1)
plt.ylim(0)
plt.show()
