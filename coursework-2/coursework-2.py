import numpy as np
import scipy as scp
from matplotlib import pyplot as plt


# =============================================================================
# Plotting Functions
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

def shooting_2d(z, theta_0, boundary, fg, fx, n_points):
    """
    The shooting algorithm. Uses an initial guess z to integrate the IVP and
    then uses a rootfinding alogirthm to find the value of z. A grid is then
    generated and the IVP is integrated again using the guess [0, z_root]
    where z_root is the root of the function phi(z).
    """

    def IVP(q, s):
        """
        Define the IVP q'(s). This is the IVP whill will be integrated to find
        the solution theta(s).
        """
        dq = np.zeros_like(q)
        dq[0] = q[1]
        dq[1] = s * fg * np.cos(q[0]) + s * fx * np.sin(q[0])
        return dq

    def phi_z(z):
        """
        Defines the function phi(z) which takes in the intial guess of z and
        creates a residual function. Using rooting finding methods on this
        function will return an appropriate value of z to use in the intial
        guess vector when integrating the IVP to find theta(s).
        """
        ivp_int_sol = scp.integrate.odeint(IVP, [theta_0, z], boundary)
        bound = ivp_int_sol[-1, 1]
        phi = bound
        return phi

    def find_z(z_guess, phi):
        """
        Find the root of the function phi(z).
        """

        z_root = scp.optimize.root(phi, z_guess).x

        return z_root

    z_root = find_z(z, phi_z)
    s, h = np.linspace(boundary[0], boundary[1], n_points, retstep=True)
    bvp_sol = scp.integrate.odeint(IVP, [theta_0, z_root], s)

    return h, bvp_sol[:, 0]


def euler_step_2d(ds, theta, x0, z0):
    """
    Compute an Euler step to determine the value of x, z at the next interval
    on the hair grid.
    """

    N = len(theta)
    xs = np.zeros(N)
    zs = np.zeros(N)

    xs[0] = x0
    zs[0] = z0

    for i in range(N-1):
        xs[i + 1] = xs[i] + ds * np.cos(theta[i])
        zs[i + 1] = zs[i] + ds * np.sin(theta[i])

    return xs, zs


def hair_locations_task1(L, R, fg, fx, theta_0):
    """
    Function to return the location of where the hairs meet the head.
    """

    n_hairs = len(theta_0)
    n_points = 100
    boundary = [0, L]

    # the sign of the value of z needs to depend on the side of the head of
    # the hair is on.
    z = 0.6
    theta_minus_90 = theta_0 - np.pi/2
    z_guess = np.zeros_like(theta_0)
    z_guess = np.sign(theta_minus_90) * z

    x_coords = np.zeros((n_hairs, n_points))
    z_coords = np.zeros((n_hairs, n_points))

    # go through each hair and shoot for a solution theta(s)
    for hair in range(n_hairs):
        # calculate the s and theta parameters for an individual hair
        h, theta_hair = shooting_2d(
            z_guess[hair], theta_0[hair], boundary, fg, fx, n_points)

        # calculate the initial conditions for the x, z coords
        x_0 = R * np.cos(theta_0[hair])
        z_0 = R * np.sin(theta_0[hair])

        # call the Euler step to calculate the x, z coordinates of the hair
        x_coords[hair, :], z_coords[hair, :] = euler_step_2d(
                h, theta_hair, x_0, z_0)

    return x_coords, z_coords


thetas = np.linspace(0, np.pi, 20)
x_coords, z_coords = hair_locations_task1(4, 10, 0.1, 0, thetas)
plot_on_circular_head(10, x_coords, z_coords)


# =============================================================================
# Task 4
# =============================================================================

def shooting_3d(z, theta_0, phi_0, boundary, fg, fx, n_points):
    """
    The shooting algorithm. Uses an initial guess z to integrate the IVP and
    then uses a rootfinding alogirthm to find the value of z. A grid is then
    generated and the IVP is integrated again using the guess [0, z_root]
    where z_root is the root of the function phi(z).
    """

    def IVP(q, s):
        """
        Define the IVP q'(s). This is the IVP whill will be integrated to find
        the solution theta(s).
        """
        dq = np.zeros_like(q)
        dq[0] = q[1]
        dq[1] = s * fg * np.cos(q[0]) + s * fx * np.cos(q[2]) * np.sin(q[0])
        dq[2] = q[3]
        dq[3] = - s * fx * np.sin(q[2]) * np.sin(q[0])
        return dq

    def phi_z(z):
        """
        Defines the function phi(z) which takes in the intial guess of z and
        creates a residual function. Using rooting finding methods on this
        function will return an appropriate value of z to use in the intial
        guess vector when integrating the IVP to find theta(s).
        """
        ivp_int_sol = scp.integrate.odeint(IVP, [theta_0, z[0], phi_0, z[1]],
                                           boundary)
        # debug this ODEINT output ################################
        bounds = np.zeros(2)
        bounds[0] = ivp_int_sol[-1, 1]
        bounds[1] = ivp_int_sol[-1, 3]
        phi = bounds
        return phi

    def find_z(z_guess, phi):
        """
        Find the root of the function phi(z).
        """

        z_root = scp.optimize.root(phi, z_guess).x

        return z_root

    z_root = find_z(z, phi_z)
    s, h = np.linspace(boundary[0], boundary[1], n_points, retstep=True)
    bvp_sol = scp.integrate.odeint(IVP, [theta_0, z_root[0], phi_0, z_root[1]],
                                   s)

    return h, bvp_sol[:, 0], bvp_sol[:, 2]


def euler_step_3d(ds, theta, phi, x0, y0, z0):
    """
    Compute an Euler step to determine the value of x, y, z at the next
    interval on the hair grid.
    """

    N = len(theta)
    xs = np.zeros(N)
    ys = np.zeros(N)
    zs = np.zeros(N)

    xs[0] = x0
    ys[0] = y0
    zs[0] = z0

    for i in range(N-1):
        xs[i + 1] = xs[i] + ds * np.cos(theta[i]) * np.cos(phi[i])
        ys[i + 1] = ys[i] + ds * -np.cos(theta[i]) * np.sin(phi[i])
        zs[i + 1] = zs[i] + ds * np.sin(theta[i])

    return xs, ys, zs


def hair_locations_task4(L, R, fg, fx, theta_0, phi_0):
    """
    Function to return the location of where the hairs meet the head.
    """

    n_hairs = len(theta_0)
    n_points = 100
    boundary = [0, L]

    # the sign of the value of z needs to depend on the side of the head of
    # the hair is on.
    z_theta = 0.003
    z_phi = 0.06
    theta_minus_45 = theta_0 - np.pi/4
    phi_minus_90 = phi_0 - np.pi/2
    z_guess = np.zeros((n_hairs, 2))
    z_guess[:, 0] = np.sign(theta_minus_45) * z_theta
    z_guess[:, 1] = np.sign(phi_minus_90) * z_phi

    x_coords = np.zeros((n_hairs, n_points))
    y_coords = np.zeros((n_hairs, n_points))
    z_coords = np.zeros((n_hairs, n_points))

    # go through each hair and shoot for a solution theta(s)
    for hair in range(n_hairs):
        # calculate the s and theta parameters for an individual hair
        h, theta_hair, phi_hair = shooting_3d(
            z_guess[hair, :], theta_0[hair], phi_0[hair], boundary, fg, fx,
            n_points)

        # calculate the initial conditions for the x, z coords
        x_0 = R * np.cos(theta_0[hair]) * np.cos(phi_0[hair])
        y_0 = - R * np.cos(theta_0[hair]) * np.sin(phi_0[hair])
        z_0 = R * np.sin(theta_0[hair])

        # call the Euler step to calculate the x, z coordinates of the hair
        x_coords[hair, :], y_coords[hair, :], z_coords[hair, :] = \
            euler_step_3d(
                h, theta_hair, phi_hair, x_0, y_0, z_0)

    return x_coords, y_coords, z_coords

theta1 = np.linspace(0, 0.49 * np.pi, 100)
phi1 = np.linspace(0, np.pi, 100)

x, y, z = hair_locations_task4(4, 10, 0.1, 0, theta1, phi1)
print(x[0:2], y[0:2], z[0:2])
