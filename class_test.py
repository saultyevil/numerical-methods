import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy import integrate, linalg


def question_1():
    """
    Solution to question 1 goes here
    """

    A = np.array([[3, 2], [4, 1]])

    return A.T


def question_2():
    """
    Solution to question 2 goes here
    """

    a = np.array([1, 2, 3])
    b = np.array([1, 0, -1])

    return np.cross(a, b)


def question_3():
    """
    Solution to question 3 goes here
    """

    diag_elements = np.array([1, 0, -1])

    return np.diag(diag_elements)


def question_4():
    """
    Solution to question 4 goes here
    """

    rand_array = np.random.rand(10, 4)

    return rand_array.shape


def question_5():
    """
    Solution to question 5 goes here
    """

    F = np.array([[1, 2], [3, 4]])

    G1 = linalg.expm(F)
    G2 = np.exp(F)

    return G1, G2


def question_6():
    """
    Solution to question 6 goes here
    """

    C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    d = np.array([-2, -2, 7])

    return np.linalg.solve(C, d)


def question_7():
    """
    Solution to question 7 goes here
    """

    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)

    X, Y = np.meshgrid(x, y)
    Z = np.cos(2*(x ** 2 + y ** 2)) * np.exp(x ** 2 - y ** 2)

    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.hot)
    ax.view_init(50, 150)
    plt.show()


def question_8():
    """
    Solution to question 8 goes here
    """

    C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    eigval, eigvec = np.linalg.eig(C)

    return np.max(np.abs(eigval))


def question_9():
    """
    Solution to question 9 goes here
    """

    def integrand1(x):

        return (np.exp(-x ** 2) * np.cos(x)) / (1 + x ** 3)

    return integrate.quad(integrand1, 0, 2)


def question_10():
    """
    Solution to question 10 goes here
    """

    fib_array = np.zeros(40)
    fib_array[0] = 1
    fib_array[1] = 1

    for step in range(1, Nsteps-1):
        fib_array[step+1] = fib_array[step] + fib_array[step-1]

    X = (1 + np.sqrt(5))/2

    plt.plot(np.log(np.abs(fib_array[1:]/fib_array[:-1] - X)))
    plt.show()


if __name__ == "__main__":
    print("Solution to question 1:")
    print(question_1())
    print("Solution to question 2:")
    print(question_2())
    print("Solution to question 3:")
    print(question_3())
    print("Solution to question 4:")
    print(question_4())
    print("Solution to question 5:")
    print(question_5())
    print("Solution to question 6:")
    print(question_6())
    print("Solution to question 7:")
    print(question_7())
    print("Solution to question 8:")
    print(question_8())
    print("Solution to question 9:")
    print(question_9())
    print("Solution to question 10:")
    print(question_10())
