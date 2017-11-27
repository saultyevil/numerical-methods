# =============================================================================
# Application of the shooting method to solve a boundary value problem
#
# The steps for this are as follows:
#   1.) Write the BVP as an IVP and guess the missing initial condition Z0
#   2.) Compute the value of y(x, z) by integrating the IVP
#   3.) Compute the residual phi(z) = y(b, zn) - B
#   4.) Repeat until phi(z) = 0
#
#   y'' + y' + 1 = 0; y(0) = 0, y(1) = 1 for x in [0, 1]
#
#   IVP : y'' + y' + 1 = 0,   y(0) = 0, y'(0) = z
#   Reformulate in first order form:
#   y1' = y2            y1(0) = 0
#   y2' = -1 - y2       y2(0) = z
#
#   Want to solve the IVP: phi(z) = y(1) - 1 = y1(1) - 1
#
# =============================================================================

import numpy as np
import scipy as scp
from matplotlib import pyplot as plt


def f(q, x):
    """
    Define the IVP problem.
    """

    dqdx = np.zeros_like(q)
    dqdx[0] = q[1]
    dqdx[1] = -1 - q[1]

    return dqdx


def shooting_ivp(z):
    """
    Integrate the IVP problem.
    """

    # find the solution to the initial value problem
    # odeint(function, initial condition, boundary to solve for)
    ivp_sol = scp.integrate.odeint(f, [0, z], [0, 1])
    # ivp sol has the form:
    # row 0: the points which the ODE was integrated at
    # row 1: the solution at these points

    y_boundary = ivp_sol[-1, 0]  # find the y value at the boundary B

    # find z so that the boundary condition y(1, z) = 1 is satisfied.
    # thus we find the root by root finding y_boundary - 1.
    # phi(z) = y(b, z) - B
    phi = y_boundary - 1  # return this function as we will be finding the
    # root of this function

    return phi


def shooting():
    """
    Employ the shooting method.
    """

    z_guess = 1.0  # initial guess for the IVP
    # compute the function phi(z) using shooting ivp, and find the root to this
    # non-linear equation using the Newton method. This will give a better
    # estimate for the value of z to use when integrating the IVP
    # newton(function, initial estimate)
    z_proper = scp.optimize.newton(shooting_ivp, z_guess)

    # construct a grid to integrate over
    x, h = np.linspace(0, 1, 50, retstep=True)
    # find the solution by integrating the IVP again with the new guess for z
    solution = scp.integrate.odeint(f, [0, z_proper], x)

    return x, solution[:, 0]

# =============================================================================
# Plot the solution and the error
# =============================================================================

# plot the solution
x, y = shooting()
plt.figure(figsize=(11, 5))
plt.plot(x, y, label='shooting')
plt.plot(x, 2 * np.exp(1)/(np.exp(1) - 1) * (1 - np.exp(-x)) - x, '+',
         label='exact')
plt.legend()
plt.xlabel(r"$x$")
plt.show()

# plot the absolute error
plt.figure(figsize=(11, 5))
plt.plot(x, y - (2 * np.exp(1)/(np.exp(1) - 1) * (1 - np.exp(-x)) - x))
plt.xlabel(r"$x$")
plt.show()
