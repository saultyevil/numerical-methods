# =============================================================================
# Implementation of the RK3 method for solving an ODE
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Task 1
# =============================================================================


def rk3(A, bvector, y0, interval, N):
    """
    Use the RK3 method to find the solution to an ODE.
    """

    # create the array of x values using the interval provided
    # set h as the step between the x values.
    x, h = np.linspace(interval[0], interval[1], N+1, retstep=True)
    y = np.zeros((len(y0), N+1))  # create array to hold y values
    y[:, 0] = y0  # assign first column of y to be initial guess

    for n in range(N):
        # calculate the k values
        k1 = y[:, n] + h * (np.dot(A, y[:, n]) + bvector(x[n]))
        k2 = (3/4) * y[:, n] + (1/4) * k1 + (1/4) * h * \
            (np.dot(A, k1) + bvector(x[n] + h))
        # calculate y_(n+1)
        y[:, n+1] = (1/3) * y[:, n] + (2/3) * k2 + (2/3) * h * \
            (np.dot(A, k2) + bvector(x[n] + h))

    return x, y


# =============================================================================
# Task 2
# =============================================================================

def dirk3(A, bvector, y0, interval, N):
    """
    Use the direct RK3 method to find to the solution to an ODE.
    """
    # create the array of x values using the interval provided
    # set h as the step between the x values.
    x, h = np.linspace(interval[0], interval[1], N+1, retstep=True)
    y = np.zeros((len(y0), N+1))  # create array to hold y values
    y[:, 0] = y0  # assign first column of y to be initial guess

    # define constants
    mu = (1/2) * (1 - 1/np.sqrt(3))
    nu = (1/2) * (np.sqrt(3) - 1)
    gamma = (3)/(2 * (3 + np.sqrt(3)))
    lamb = (3 * (1 + np.sqrt(3)))/(2 * (3 + np.sqrt(3)))


    return x, y

# =============================================================================
# Task 3
# =============================================================================

a1 = 1000
a2 = 1
A = np.array([[-a1, 0], [a1, -a2]])
x, y = rk3(A, lambda x: 0, [1, 0], [0, 0.1], 400)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax1.semilogy(x, y[0, :])
ax1.set_xlim(0, 0.1)
ax2 = fig.add_subplot(122)
ax2.plot(x, y[1, :])
ax2.set_xlim(0, 0.1)
plt.show()
