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
    Use the explicit RK3 method to find the solution of an ODE.

    This function solves for ODEs in the form:
            y' = Ay + b(x),
    where A is a matrix of coefficents and b is a function of x.

    Parameters
    ----------
    A: array. The matrix of coefficents for the linear equation which
              represents the ODE.
    bvector: function. A function which defines the function in the linear
                       equation which represents the ODE.
    y0: list. The initial guess for the employed method. The format is
              expected to be a list with each element being for a different
              component for y0, i.e. y0 = [1, 0]
    interval: list. A list containing two values for the interval to iterate
                    over. For example, for an interval of 0 to 0.1, use
                    interval = [0, 0.1].
    N: integer. The number of sampling points to use in the interval.

    Returns
    -------
    x: array. The x values used to compute the approximation.
    y: array. The approximated values of y from the RK3 method.
    """

    assert(N > 0), 'Sampling points needs to be more than zore.'
    assert(interval[1] > interval[0]), 'Incorrect interval used.'
    assert(len(A) == len(y0)), 'Different sizes for matrix A and vector y0.'

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
    Use the diagonally implicit RK3 method to find to the solution of an ODE.

    This function solves for ODEs in the form:
            y' = Ay + b(x),
    where A is a matrix of coefficents and b is a function of x.

    Parameters
    ----------
    A: array. The matrix of coefficents for the linear equation which
              represents the ODE.
    bvector: function. A function which defines the function in the linear
                       equation which represents the ODE.
    y0: list. The initial guess for the employed method. The format is
              expected to be a list with each element being for a different
              component for y0, i.e. y0 = [1, 0]
    interval: list. A list containing two values for the interval to iterate
                    over. For example, for an interval of 0 to 0.1, use
                    interval = [0, 0.1].
    N: integer. The number of sampling points to use in the interval.

    Returns
    -------
    x: array. The x values used to compute the approximation.
    y: array. The approximated values of y from the DIRK3 method.
    """

    assert(N > 0), 'Sampling points needs to be more than zore.'
    assert(interval[1] > interval[0]), 'Incorrect interval used.'
    assert(len(A) == len(y0)), 'Different sizes for matrix A and vector y0.'

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

    ident = np.identity(len(A)) - h * mu * A

    for n in range(N):
        # calculate the k values
        k1 = np.linalg.solve(ident, y[:, n] + h * mu * bvector(x[n] + h * mu))
        k2 = np.linalg.solve(ident, k1 + h * nu * (np.dot(A, k1) +
                             bvector(x[n] + h * mu)) + h * mu *
                             bvector([x[n] + h * nu + 2 * h * mu]))
        # calculate y_n+1
        y[:, n+1] = (1 - lamb) * y[:, n] + lamb * k2 + h * gamma * \
            (np.dot(A, k2) + bvector(x[n] + h * nu + 2 * h * mu))

    return x, y


# =============================================================================
# Task 3
# =============================================================================

# define the coefficent matrix, interval and initial value
a1 = 1000
a2 = 1
A = np.array([[-a1, 0], [a1, -a2]])
y0 = [1, 0]
interval = [0, 0.1]

# arrays for storing h value and error
error_rk3 = np.zeros((2, 10))
error_dirk3 = np.zeros((2, 10))

for i, k in enumerate(np.arange(1, 11)):  # loop over the range of k values
    N = 40 * k
    # compute h and store it in the errors array
    h = (interval[1] - interval[0])/N
    error_rk3[0, i] = h
    error_dirk3[0, i] = h

    # compute y approximations using rk3 and dirk3 algorithms
    x_rk3, y_rk3 = rk3(A, lambda x: 0, y0, interval, N)
    x_dirk3, y_dirk3 = dirk3(A, lambda x: 0, y0, interval, N)
    # calc exact using the same x's from the rk3 algorithm
    # x_rk3 and x_dirk3 will be the same!
    y_exact = np.array([[np.exp(-a1 * x_rk3)],
                        [(a1/(a1-a2)) * (np.exp(-a2 * x_rk3) -
                         np.exp(-a1 * x_rk3))]])
    # panic when it comes out 3d and then use np.reshape to make it better
    y_exact = np.reshape(y_exact, (2, N+1))

    # calculate the error norms
    error_rk3[1, i] = h * np.sum(np.abs((y_rk3[1, 1:] - y_exact[1, 1:]) /
                                        y_exact[1, 1:]))
    error_dirk3[1, i] = h * np.sum(np.abs((y_dirk3[1, 1:] - y_exact[1, 1:]) /
                                          y_exact[1, 1:]))


# compute the polyfits
# if error = A * h^s, then linear line best fit of log(error) v log(h) will
# have the slope s, e.g. the order of convergence
rk3_polyfit = np.polyfit(np.log(error_rk3[0, :-1]),
                         np.log(error_rk3[1, :-1]), 1)
dirk3_polyfit = np.polyfit(np.log(error_dirk3[0, :-1]),
                           np.log(error_dirk3[1, :-1]), 1)

# plot the results!
fig_errors = plt.figure(figsize=(15, 6))

# plot the errors v h =========================================================
ax1 = fig_errors.add_subplot(121)
ax1.loglog(error_rk3[0, :], error_rk3[1, :], 'kx')
ax1.loglog(error_rk3[0, :],
           np.exp(rk3_polyfit[1]) * error_rk3[0, :] ** (rk3_polyfit[0]),
           label='RK3 line gradient: {:6.4f}'.format(rk3_polyfit[0]))
ax1.set_xlabel('h')
ax1.set_ylabel('error')
ax1.legend()

ax2 = fig_errors.add_subplot(122)
ax2.loglog(error_dirk3[0, :], error_dirk3[1, :], 'kx')
ax2.loglog(error_dirk3[0, :],
           np.exp(dirk3_polyfit[1]) * error_dirk3[0, :] ** (dirk3_polyfit[0]),
           label='DIRK3 line gradient: {:6.4f}'.format(dirk3_polyfit[0]))
ax2.set_xlabel('h')
ax2.set_ylabel('error')
ax2.legend()

plt.savefig('task3_errors.pdf')
plt.show()

# plot the algorithm against the exact solution ===============================
fig_comparison = plt.figure(figsize=(15, 12))

ax3 = fig_comparison.add_subplot(221)
ax3.semilogy(x_rk3, y_rk3[0, :], label='rk3 y1')
ax3.semilogy(x_rk3, y_exact[0, :], '--', label='exact y1')
ax3.legend()

ax4 = fig_comparison.add_subplot(222)
ax4.plot(x_rk3, y_rk3[1, :], label='rk3 y2')
ax4.plot(x_rk3, y_exact[1, :], '--', label='exact y2')
ax4.legend()

ax5 = fig_comparison.add_subplot(223)
ax5.semilogy(x_dirk3, y_dirk3[0, :], label='dirk3 y1')
ax5.semilogy(x_dirk3, y_exact[0, :], '--', label='exact y1')
ax5.legend()

ax6 = fig_comparison.add_subplot(224)
ax6.plot(x_dirk3, y_dirk3[1, :], label='dirk3 y2')
ax6.plot(x_dirk3, y_exact[1, :], '--', label='exact y2')
ax6.legend()

plt.savefig('task3_comparison.pdf')
plt.show()

# =============================================================================
# Task 4
# =============================================================================
