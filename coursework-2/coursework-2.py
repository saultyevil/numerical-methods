import numpy as np
import scipy as scp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# Simulation Parameters
# =============================================================================

# parameters which are constant throughout

L = 4       # length of hair in cm
R = 10      # radius of head in cm
fg = 0.1    # force from gravity in cm ** -3


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_on_circular_head(R, x_hair, z_hair, fg, fx):
    """
    Plot a circular head of radius R and plot the hair positions on the head.

    Parmeters
    ---------
    R: float.
        The radius of the head in units of cm.
    x_hair: n_hairs x n_points array of floats.
        The x coordinates of each hair.
    z_hair: n_hairs x n_points array of floats.
        the z coordinates of each hair.
    fg: float.
        The value of the force acting upon the hair due to gravity. In units of
        cm ** -3.
    fx: float.
        The value of the forced acting upon the hair due to wind in the
        positive x-direction. In units of

    Returns
    -------
    A plot showing the head and the positions of the hairs. Printed both to the
    console window and saved to the working directory.
    """

    # generate the coordinates where the hair will be placed, this will create
    # a semi-circle for the top half of the head. Plot -z for the bottom half
    # of the head, but don't plot hair as it doesn't grow here because we are
    # not modelling beards
    x = np.linspace(-R, R, 1000)
    z = np.sqrt(-x ** 2 + R ** 2)

    n_hairs = x_hair.shape[0]

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.set_xlim(-R - 5, R + 5)
    ax1.set_ylim(-R - 5, R + 5)
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$z$")

    # plot the head by plotting two semi-circles
    ax1.plot(x, z, 'b-')
    ax1.plot(x, -z, 'b-')

    # plot the individual hairs, one at a time so I can plot as a line rather
    # than as x's or o's
    for hair in range(n_hairs):
        ax1.plot(x_hair[hair, :], z_hair[hair, :], '-')
        # no colour specification makes zany multi-coloured hair

    plt.gca().set_aspect('equal')
    plt.savefig('hair_phi=0__fg={}_fx={}.pdf'.format(fg, fx))
    plt.show()


def plot_on_circular_head_3d(R, x_hair, y_hair, z_hair, fg, fx):
    """
    Plot a circular head of radius R and plot the hair positions on the head.

    Parmeters
    ---------
    R: float.
        The radius of the head in units of cm.
    x_hair: n_hairs x n_points array of floats.
        The x coordinates of each hair.
    y_hair: n_hairs x n_points array of floats.
        The x coordinates of each hair.
    z_hair: n_hairs x n_points array of floats.
        the z coordinates of each hair.
    fg: float.
        The value of the force acting upon the hair due to gravity. In units of
        cm ** -3.
    fx: float.
        The value of the forced acting upon the hair due to wind in the
        positive x-direction. In units of

    Returns
    -------
    A plot showing the head and the positions of the hairs in the x, z and the
    y, z planes. Printed both to the console window and saved to the working
    directory.
    """

    # generate the coordinates where the hair will be placed, this will create
    # a semi-circle
    x = np.linspace(-R, R, 1000)
    y = np.linspace(-R, R, 1000)
    z = np.sqrt(-x ** 2 + R ** 2)

    n_hairs = x_hair.shape[0]

    fig = plt.figure(figsize=(24, 6))

    # subplot for x, z plane
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(-R - 5, R + 5)
    ax1.set_ylim(-R - 5, R + 5)
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$z$")
    plt.gca().set_aspect('equal')
    # subplot for y, z plane
    ax2 = fig.add_subplot(122)
    ax2.set_xlim(-R - 5, R + 5)
    ax2.set_ylim(-R - 5, R + 5)
    ax2.set_xlabel(r"$y$")
    ax2.set_ylabel(r"$z$")
    plt.gca().set_aspect('equal')  # I seem to need to do this twice

    # plot a circle for the 2D head slice. No heads were harmed in the
    # making of these head slices
    ax1.plot(x, z, 'b-')
    ax1.plot(x, -z, 'b-')
    ax2.plot(y, z, 'b-')
    ax2.plot(y, -z, 'b-')

    # plot the individual hairs
    for hair in range(n_hairs):
        ax1.plot(x_hair[hair, :], z_hair[hair, :], 'k-')
        ax2.plot(y_hair[hair, :], z_hair[hair, :], 'k-')

    plt.savefig('hair_3d_slices_fg={}_fx={}.pdf'.format(fg, fx))
    plt.show()


def plot_on_sphereical_head(R, x_hair, y_hair, z_hair, fg, fx):
    """
    Plot a spherical head of radius R and plot the hair positions on the head.

    Parmeters
    ---------
    R: float.
        The radius of the head in units of cm.
    x_hair: n_hairs x n_points array of floats.
        The x coordinates of each hair.
    y_hair: n_hairs x n_points array of floats.
        The x coordinates of each hair.
    z_hair: n_hairs x n_points array of floats.
        the z coordinates of each hair.
    fg: float.
        The value of the force acting upon the hair due to gravity. In units of
        cm ** -3.
    fx: float.
        The value of the forced acting upon the hair due to wind in the
        positive x-direction. In units of

    Returns
    -------
    A plot showing the head and the positions of the hairs.Printed both to the
    console window and saved to the working directory.
    """

    n_hairs = x_hair.shape[0]

    # generate theta and phi values for a sphere
    theta = np.linspace(0, 2 * np.pi, 1000)
    phi = np.linspace(0, 2 * np.pi, 1000)

    # create a meshgrid of the theta and phi values
    PHI, THETA = np.meshgrid(theta, phi)

    # calculate the x, y, z coordinates for the sphere
    x = R * np.sin(PHI) * np.cos(THETA)
    y = R * np.sin(PHI) * np.sin(THETA)
    z = R * np.cos(PHI)

    fig = plt.figure(figsize=(12, 12))

    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlim(-R - 5, R + 5)
    ax1.set_ylim(-R - 5, R + 5)
    ax1.set_zlim(-R - 5, R + 5)
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')
    ax1.set_zlabel(r'$z$')
    plt.gca().set_aspect('equal')

    # plot the sphere using the carterisan coordinates
    ax1.plot_surface(x, y, z)

    # plot each hair individually so I can use a line instead of x's or o's
    for hair in range(n_hairs):
        ax1.scatter(x_hair[hair, :], y_hair[hair, :], z_hair[hair, :], '-')

    plt.savefig('hair_3d_fg={}, fx={}.pdf'.format(fg, fx))
    plt.show()


# =============================================================================
# Task 1 Functions
# =============================================================================

def shooting_2d(z, theta_0, boundary, fg, fx, n_points, root_return=False):
    """
    The shooting algorithm. Uses an initial guess z to integrate the IVP and
    then uses a rootfinding alogirthm to find the value of z. A grid is then
    generated and the IVP is integrated again using the guess [0, z_root]
    where z_root is the root of the function phi(z) to calculate the solution
    of the BVP.

    Parameters
    ----------
    z: float.
        The value of the initial guess for theta prime at s = 0.
    theta_0: 1 x n_hairs array of floats.
        The initial lattitude angle of each hair.
    boundary: 1 x 2 array of floats.
        An array containing the left hand side and right hand side boundary for
        the problem. Given in cartesian coordinates in units of cm.
    fg: float.
        The value of the force acting upon the hair due to gravity. In units of
        cm ** -3.
    fx: float.
        The value of the forced acting upon the hair due to wind in the
        positive x-direction. In units of
    n_points: float.
        The number of points to be used on the grid to integrate the IVP.
    root_return: boolean.
        If True, the function will only return the root to the function phi(z)
        defined in the function phi_z. If False (the default value), the
        function will return the grid spacing on the hair, ds, and the solution
        to the BVP.

    Returns
    -------
    h: float, if return_root = False.
        The value of the spacing on the grid used to integrate the IVP to find
        the solution to the BVP.
    bvp_sol: 1 x n_points array of floats, if return_root = False.
        The solution to the BVP for each points on the grid.
    z_root: float, if return_root = True.
        The root of phi(z), i.e. the value of the theta prime at s = 0.
    """

    def IVP(q, s):
        """
        Define the IVP q'(s). This is the IVP whill will be integrated to find
        the solution theta(s).

        Parameters
        ----------
        q: 1 x 2 array of floats.
            The vector of initial guesses used to integrate the IVP.
        s: 1 x n_points array of floats.
            An array containing the grid positions used to integrate the IVP.

        Returns
        -------
        dq: 1 x 2 array of floats.
            The definition of the IVP.
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

        Parameters
        ----------
        z: float.
            The value of the initial guess for theta prime at s = 0.

        Returns
        -------
        phi: float.
            The value of phi(z) calculated for the current value of z. Use a
            root finding method to find the value of z such that phi(z) = 0 to
            solve the BVP.
        """
        ivp_int_sol = scp.integrate.odeint(IVP, [theta_0, z], boundary,
                                           full_output=1)

        # if the integration is not successful, then print that to the screen
        if ivp_int_sol[1]['message'] != 'Integration successful.':
            print(ivp_int_sol[1]['message'])

        # phi(z) = theta'(L, z) - L' = theta_2(L, z)
        bound = ivp_int_sol[0][-1, 1]
        phi = bound

        return phi

    def find_z(z_guess, phi):
        """
        Find the root of the function phi(z).

        Parameters
        ----------
        z_guess: float.
            The value of the initial guess for theta prime at s = 0.
        phi: function.
            The function to root find, i.e. phi(z).

        Returns
        -------
        z_root: float.
            The value of z which is the root of phi(z).
        """

        z_root = scp.optimize.root(phi, z_guess).x

        return z_root

    # if reoot_return is True, return just the value of the root, i.e. the
    # value of theta prime at s = 0, of a hair
    if root_return is False:
        # returns the ds separation, h, and the theta solution on s
        z_root = find_z(z, phi_z)  # find the value of z to use for theta'(0)
        # create the integration grid and and find spacing on the grid
        s, h = np.linspace(boundary[0], boundary[1], n_points, retstep=True)
        # solve the bvp
        bvp_sol = scp.integrate.odeint(IVP, [theta_0, z_root], s)
        # return theta_1 as theta_1 = theta, i.e. the solution to the BVP

        return h, bvp_sol[:, 0]

    else:
        # returns the theta prime (s=0) value
        z_root = find_z(z, phi_z)

        return z_root


def euler_step_2d(ds, theta, x0, z0):
    """
    Compute an Euler step to determine the value of x, z at the next interval
    on the hair grid.

    Parameters
    ----------
    ds: float.
        The separation of the grid used to integrate the IVP.
    theta: 1 x n_points array of floats.
        The value of theta at each point on the integration grid.
    x0: float.
        The x coordinate of the hair's initial lattitude position.
    z0: float.
        The z coordinate of the hair's initiual lattitude position.

    Returns
    -------
    x, z: 1 x n_points array of floats.
        The location a hair on a cartesian grid.
    """

    N = len(theta)
    xs = np.zeros(N)
    zs = np.zeros(N)

    # set the initial values for x and z
    xs[0] = x0
    zs[0] = z0

    # iterate through all the points on the grid to caculate the value of
    # x, z
    for i in range(N-1):
        xs[i + 1] = xs[i] + ds * np.cos(theta[i])
        zs[i + 1] = zs[i] + ds * np.sin(theta[i])

    return xs, zs


def hair_locations_2d(L, R, fg, fx, theta_0):
    """
    Function to return the location of hairs on a head.

    Parameters
    ----------
    L: float.
        The size of each hair in units of cm.
    R: float.
        The radius of the head in units of cm.
    fg: float.
        The value of the force acting upon the hair due to gravity. In units of
        cm ** -3.
    fx: float.
        The value of the forced acting upon the hair due to wind in the
        positive x-direction. In units of
    theta_0: 1 x n_hairs array of floats.
        The initial lattitude angle of each hair.

    Returns
    -------
    x_coords, z_coords: n_hairs x n_points array of floats.
        The x, z coordinates of each hair.
    """

    n_hairs = len(theta_0)
    n_points = 100
    boundary = [0, L]

    # the sign of the value of z needs to depend on the side of the head of
    # the hair is on
    z = 0.5  # using 0.1 makes the middle stick up, v interesting
    theta_minus_90 = theta_0 - np.pi/2
    z_guess = np.zeros_like(theta_0)
    z_guess = np.sign(theta_minus_90) * z

    # use absolute value as -fx would be wind in the negative x direction
    if np.abs(fx) > 0:
        # if there is wind, use the root for the no wind case as the initial
        # guess for the wind case.
        print('Calculating initial guess values using no wind...')
        for hair in range(n_hairs):
            # iterate through the hairs and find the theta prime value used
            # for each hair in the no wind hair
            z_guess_wind = shooting_2d(
                z_guess[hair], theta_0[hair], boundary, fg, fx, n_points,
                root_return=True)
            z_guess[hair] = z_guess_wind

    x_coords = np.zeros((n_hairs, n_points))
    z_coords = np.zeros((n_hairs, n_points))

    # go through each hair and shoot for a solution theta(s)
    for hair in range(n_hairs):
        # calculate ds (the grid spacing on the hair) and theta(s) for each
        # individual hair
        h, theta_hair = shooting_2d(
            z_guess[hair], theta_0[hair], boundary, fg, fx, n_points)

        # calculate the initial conditions for the x, z coords for the hair
        x_0 = R * np.cos(theta_0[hair])
        z_0 = R * np.sin(theta_0[hair])

        # call the Euler step to calculate the x, z coordinates of the hair
        x_coords[hair, :], z_coords[hair, :] = euler_step_2d(
                h, theta_hair, x_0, z_0)

    return x_coords, z_coords


# generate the theta 0 values
n_hairs = 20
thetas = np.linspace(0, np.pi, n_hairs)

fx = 0  # no wind
x_coords, z_coords = hair_locations_2d(L, R, fg, fx, thetas)
plot_on_circular_head(R, x_coords, z_coords, fg, fx)

fx = 0.1  # introduce a bit of wind
x_coords, z_coords = hair_locations_2d(L, R, fg, fx, thetas)
plot_on_circular_head(R, x_coords, z_coords, fg, fx)


# =============================================================================
# Task 4
# =============================================================================

def shooting_3d(z, theta_0, phi_0, boundary, fg, fx, n_points,
                root_return=False):
    """
    The shooting algorithm. Uses an initial guess z to integrate the IVP and
    then uses a rootfinding alogirthm to find the value of z. A grid is then
    generated and the IVP is integrated again using the guess [0, z_root]
    where z_root is the root of the function phi(z) to calculate the solution
    of the BVP.

    Parameters
    ----------
    z: float.
        The value of the initial guess for theta prime at s = 0.
    theta_0: n x n array of floats, where n x n = n_hairs.
        The initial lattitude angle of each hair. Theta_0 needs to be part of a
        meshgrid of theta_0 and phi_0.
    phi_0: n x n array of floats, where n x n = n_hairs.
        The initial longitude angle of each hair. Phi_0 needs to be part of a
        meshgrid of theta_0 and phi_0.
    boundary: 1 x 2 array of floats.
        An array containing the left hand side and right hand side boundary for
        the problem. Given in cartesian coordinates in units of cm.
    fg: float.
        The value of the force acting upon the hair due to gravity. In units of
        cm ** -3.
    fx: float.
        The value of the forced acting upon the hair due to wind in the
        positive x-direction. In units of
    n_points: float.
        The number of points to be used on the grid to integrate the IVP.
    root_return: boolean.
        If True, the function will only return the root to the function phi(z)
        defined in the function phi_z. If False (the default value), the
        function will return the grid spacing on the hair, ds, and the solution
        to the BVP.

    Returns
    -------
    h: float, if return_root = False.
        The value of the spacing on the grid used to integrate the IVP to find
        the solution to the BVP.
    bvp_sol: 1 x n_points array of floats, if return_root = False.
        The solution to the BVP for each points on the grid for theta and phi
        separately.
    z_root: float, if return_root = True.
        The root of phi(z), i.e. the value of the theta prime at s = 0.
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
        dq[3] = - s * fx * np.sin(q[0]) * np.sin(q[2])
        return dq

    def phi_z(z):
        """
        Defines the function phi(z) which takes in the intial guess of z and
        creates a residual function. Using rooting finding methods on this
        function will return an appropriate value of z to use in the intial
        guess vector when integrating the IVP to find theta(s), phi(s).

        Parameters
        ----------
        z: float.
            The value of the initial guess for theta prime at s = 0.

        Returns
        -------
        phi: float.
            The value of phi(z) calculated for the current value of z. Use a
            root finding method to find the value of z such that phi(z) = 0 to
            solve the BVP.
        """
        ivp_int_sol = scp.integrate.odeint(IVP, [theta_0, z[0], phi_0, z[1]],
                                           boundary, full_output=1)

        # if the integration is not successful, then print that to the screen
        if ivp_int_sol[1]['message'] != 'Integration successful.':
            print(ivp_int_sol[1]['message'])

        bounds = np.zeros(2)
        bounds[0] = ivp_int_sol[0][-1, 1]
        bounds[1] = ivp_int_sol[0][-1, 3]
        phi = bounds

        return phi

    def find_z(z_guess, phi):
        """
        Find the root of the function phi(z).

        Parameters
        ----------
        z_guess: float.
            The value of the initial guess for theta prime at s = 0.
        phi: function.
            The function to root find, i.e. phi(z).

        Returns
        -------
        z_root: float.
            The value of z which is the root of phi(z).
        """

        z_root = scp.optimize.root(phi, z_guess).x

        return z_root

    # if reoot_return is True, return just the value of the root, i.e. the
    # value of theta prime at s = 0, of a hair
    if root_return is False:
        # returns the ds separation, h, and the theta solution on s
        z_root = find_z(z, phi_z)  # find the values of z to use
        # create the integration grid and and find spacing on the grid
        s, h = np.linspace(boundary[0], boundary[1], n_points, retstep=True)
        bvp_sol = scp.integrate.odeint(IVP, [theta_0, z_root[0], phi_0,
                                             z_root[1]], s)
        # return theta_1 and phi_1, i.e. theta and phi

        return h, bvp_sol[:, 0], bvp_sol[:, 2]

    else:
        # returns the theta', phi' at s = 0 value
        z_root = find_z(z, phi_z)

        return z_root


def euler_step_3d(ds, theta, phi, x0, y0, z0):
    """
    Compute an Euler step to determine the value of x, y, z at the next
    interval on the hair grid.

    Parameters
    ----------
    ds: float.
        The separation of the grid used to integrate the IVP.
    theta: 1 x n_points array of floats.
        The value of theta at each point on the integration grid.
    x0: float.
        The x coordinate of the hair's initial lattitude and longitude
        position.
    y0: float.
        The y coordinate of the hair's initial lattitude and longitude
        position.
    z0: float.
        The z coordinate of the hair's initial lattitude and longitude
        position.

    Returns
    -------
    x, y, z: 1 x n_points array of floats.
        The location a hair on a cartesian grid.
    """

    N = len(theta)
    xs = np.zeros(N)
    ys = np.zeros(N)
    zs = np.zeros(N)

    # set the initial values for x and z
    xs[0] = x0
    ys[0] = y0
    zs[0] = z0

    # iterate through all the points on the grid to caculate the value of
    # x, y, z
    for i in range(N-1):
        xs[i + 1] = xs[i] + ds * np.cos(theta[i]) * np.cos(phi[i])
        ys[i + 1] = ys[i] + ds * -np.cos(theta[i]) * np.sin(phi[i])
        zs[i + 1] = zs[i] + ds * np.sin(theta[i])

    return xs, ys, zs


def hair_locations_3d(L, R, fg, fx, theta_0, phi_0):
    """
    Function to return the location of hairs on a head.

    Parameters
    ----------
    L: float.
        The size of each hair in units of cm.
    R: float.
        The radius of the head in units of cm.
    fg: float.
        The value of the force acting upon the hair due to gravity. In units of
        cm ** -3.
    fx: float.
        The value of the forced acting upon the hair due to wind in the
        positive x-direction. In units of
    theta_0: n x n array of floats, where n x n = n_hairs.
        The initial lattitude angle of each hair. Theta_0 needs to be part of a
        meshgrid of theta_0 and phi_0.
    phi_0: n x n array of floats, where n x n = n_hairs.
        The initial longitude angle of each hair. Phi_0 needs to be part of a
        meshgrid of theta_0 and phi_0.
    Returns
    -------
    x_coords, y_coords, z_coords: n_hairs x n_points array of floats.
        The x, z coordinates of each hair.
    """

    n_grid_theta = theta_0.shape[0]
    n_grid_phi = phi_0.shape[0]
    n_hairs = n_grid_theta * n_grid_phi
    n_points = 100
    boundary = [0, L]

    z_theta = 0.01
    z_phi = 0.1
    z_guess = np.zeros((2, n_grid_theta, n_grid_phi))
    z_guess[:, :, :] = z_theta

    if np.abs(fx) > 0:
        # if there is wind, use the root for the no wind case as the initial
        # guess for the wind case.
        print('Calculating initial guess values using no wind...')
        for i in range(n_grid_theta - 1):
            for j in range(n_grid_phi - 1):
                # create a list for the z guess at the location theta, phi
                z_guesses = [z_guess[0, i, j], z_guess[1, i, j]]
                z_guess_wind = shooting_3d(
                    z_guesses, theta_0[i, j], phi_0[i, j], boundary, fg,
                    fx, n_points, root_return=True)
                z_guess[i, j] = z_guess_wind

    x_coords = np.zeros((n_hairs, n_points))
    y_coords = np.zeros((n_hairs, n_points))
    z_coords = np.zeros((n_hairs, n_points))

    # go through each hair and shoot for a solution theta(s)
    hair = 0  # counter variable to track the hair we are on
    for i in range(n_grid_theta - 1):
        for j in range(n_grid_phi - 1):
            # create a list for the z guess at the location theta, phi
            z_guesses = [z_guess[0, i, j], z_guess[1, i, j]]
            # calculate the s and theta parameters for an individual hair
            h, theta_hair, phi_hair = shooting_3d(
                z_guesses, theta_0[i, j], phi_0[i, j], boundary, fg, fx,
                n_points)

            # calculate the initial conditions for the x, z coords
            x_0 = R * np.cos(theta_0[i, j]) * np.cos(phi_0[i, j])
            y_0 = - R * np.cos(theta_0[i, j]) * np.sin(phi_0[i, j])
            z_0 = R * np.sin(theta_0[i, j])

            # call the Euler step to calculate the x, z coordinates of the hair
            x_coords[hair, :], y_coords[hair, :], z_coords[hair, :] = \
                euler_step_3d(
                    h, theta_hair, phi_hair, x_0, y_0, z_0)

            # increment the hair counter, so it goes to the next hair on the
            # next loop of the grid
            hair += 1

    return x_coords, y_coords, z_coords

# create the mesh grid of theta and phi positions for the hair locations
thetas = np.linspace(0, 0.49 * np.pi, 10)
phis = np.linspace(0, np.pi, 10)
THETA, PHI = np.meshgrid(thetas, phis)

fx = 0  # use this for testing purposes
# fx = 0.05  # just a little bit of wind
x_coords, y_coords, z_coords = hair_locations_3d(L, R, fg, fx, THETA, PHI)
plot_on_circular_head_3d(R, x_coords, y_coords, z_coords, fg, fx)
plot_on_sphereical_head(R, x_coords, y_coords, z_coords, fg, fx)
