import numpy as np
import scipy as scp
from matplotlib import pyplot as plt


# =============================================================================
# Functions to Plot the Head and Hair
# =============================================================================

def plot_on_circular_head(R, x_hair, z_hair):
    """
    Create a circle of radius R and plot the hair.
    """

    # generate the coordinates where the hair will be placed, this will create
    # a semi-circle, but hair doesn't grow here because we are not modelling
    # beards in this work
    x = np.linspace(-R, R, 1000)
    z = np.sqrt(-x ** 2 + R ** 2)

    n_hairs = x_hair.shape[0]

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.set_xlim(-15, 15)
    ax1.set_ylim(-15, 15)
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$z$")
    # plot the circle by plotting two semi-circles
    ax1.plot(x, z, 'b-')
    ax1.plot(x, -z, 'b-')
    # plot the hair
    for i in range(n_hairs):
        ax1.plot(x_hair[i, :], z_hair[i, :], 'k-')

    plt.gca().set_aspect('equal')
    plt.savefig('hair_phi=0.pdf')
    plt.show()


def sphereical_head(R):
    """
    Create a sphere of radius R and plot the hair.

    x ** 2 + y ** 2 + z ** 2 = R ** 2
    """

    theta = np.linspace(0, 10, 1000)
    phi = np.linspace(0, 10, 1000)

    x = theta
    y = theta
    z = phi

    return x, y, z

# =============================================================================
# Task 1 Functions
# =============================================================================


def IVP_phi_0(q, s, fg, fx):
    """
    Define the IVP q'(s). This is the IVP whill will be integrated to find the
    solution theta(s).
    """
    dq = np.zeros_like(q)
    dq[0] = q[1]
    dq[1] = s * fg * np.cos(q[0]) + s * fx * np.sin(q[0])
    return dq


def func_to_root_find(z, theta_0, boundary, fg, fx):
    """
    Defines the function phi(z) which takes in the intial guess of z and
    creates a residual function. Using rooting finding methods on this function
    will return an appropriate value of z to use in the intial guess vector
    when integrating the IVP to find theta(s).
    """
    ivp_int_sol = scp.integrate.odeint(IVP_phi_0, [theta_0, z], boundary,
                                       args=(fg, fx))
    bound = ivp_int_sol[-1, 1]
    phi = bound
    return phi


def shooting_method(z, theta_0, boundary, fg, fx, n_points):
    """
    The shooting algorithm. Uses an initial guess z to integrate the IVP and
    then uses a rootfinding alogirthm to find the value of z. A grid is then
    generated and the IVP is integrated again using the guess [0, z_root]
    where z_root is the root of the function phi(z).
    """

    z_guess = z
    z_root = scp.optimize.root(func_to_root_find, z_guess,
                               args=(theta_0, boundary, fg, fx)).x

    s, h = np.linspace(boundary[0], boundary[1], n_points, retstep=True)
    bvp_sol = scp.integrate.odeint(IVP_phi_0, [0, z_root], s,
                                   args=(fg, fx))

    return s, bvp_sol[:, 0]


def hair_locations_phi_0(L, R, fg, fx, theta_0):
    """
    Function to return the location of where the hairs meet the head.
    """

    n_hairs = len(theta_0)
    n_points = 50
    boundary = [0, L]

    z = 0.2  # the value for the initial guess used in the shooting method

    theta_hair = np.zeros((n_hairs, n_points))
    s_hair = np.zeros((n_hairs, n_points))
    # go through each hair and shoot for a solution theta(s)
    for hair in range(n_hairs):
        s, theta = shooting_method(z, theta_0[hair], boundary, fg, fx,
                                   n_points)
        theta_hair[hair, :] = theta
        s_hair[hair, :] = s

    return s_hair, theta_hair


theta = np.linspace(0, np.pi, 2)
s_hair, theta_hair = hair_locations_phi_0(4, 10, 0.1, 0, theta)
