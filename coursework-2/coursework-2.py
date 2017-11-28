import numpy as np
from matplotlib import pyplot as plt


# =============================================================================
# Task 1
# =============================================================================

def plot_on_circular_head(R, x_hair, z_hair):
    """
    Create a circle of radius R.
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
    Create a sphere of radius R.

    x ** 2 + y ** 2 + z ** 2 = R ** 2
    """

    theta = np.linspace(0, 10, 1000)
    phi = np.linspace(0, 10, 1000)

#    x =
#    y =
#    z =


def hair_locations_phi_0(L, R, fx, fg, theta):
    """
    Function to return the location of where the hairs meet the head.
    """

    n_hairs = len(theta)
    n_points = 50

    # create arrays to store the hair locations, no y coodinates as phi(s) = 0
    x_hair = np.zeros((n_hairs, n_points))
    z_hair = np.zeros((n_hairs, n_points))

    # set the original hair locations
    x_hair[:, 0] = R * np.cos(theta)
    z_hair[:, 0] = R * np.sin(theta)

    return x_hair, z_hair


theta = np.linspace(0, np.pi, 20)
x, z = hair_locations_phi_0(4, 10, 0.1, 0.3, theta)

plot_on_circular_head(10, x, z)
