"""
    Module lab2 - Linear regression, gradient descent, polynomial regression
"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
import random


# generates x and y numpy arrays for
# y = a*x + b + a * noise
# in range -1 .. 1
# with random noise of given amplitude (noise)
# visualizes it and unloads to csv
def generate_linear(a, b, noise, filename, size=100):
    # print('Generating random data y = a*x + b')
    x = 2 * np.random.rand(size, 1) - 1
    y = a * x + b + noise * a * (np.random.rand(size, 1) - 0.5)
    data = np.hstack((x, y))
    np.savetxt(filename, data, delimiter=',')
    return x, y


# thats an example of linear regression using polyfit
def linear_regression_numpy(filename):
    # now let's read it back
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    # split to initial arrays
    x, y = np.hsplit(data, 2)
    # printing shapes is useful for debugging
    print(np.shape(x))
    print(np.shape(y))
    # our model
    time_start = time()
    model = np.polyfit(np.transpose(x)[0], np.transpose(y)[0], 1)
    time_end = time()
    print(f"polyfit in {time_end - time_start} seconds")
    # our hypothesis for give x
    h = model[0] * x + model[1]
    # and check if it's ok
    plt.title("Linear regression task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    plt.plot(x, h, "r", label='model')
    plt.legend()
    plt.show()
    return model


def linear_regression_exact(filename):
    print("Ex1: your code here - exact solution using invert matrix")
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)
    x_with_ones = np.hstack([np.ones((len(x), 1)), x])
    trans_x = x_with_ones.transpose()
    xt_x_minus_one = np.linalg.pinv(trans_x.dot(x_with_ones))
    theta = xt_x_minus_one.dot(trans_x).dot(y)
    h = theta[1] * x + theta[0]
    plt.title("Linear regression task exact")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    plt.plot(x, h, "r", label='model')
    plt.legend()
    plt.show()
    theta_trans = theta.transpose()  # transpose theta to match dimension
    return theta_trans[0]


def check(model, ground_truth):
    if len(model) != len(ground_truth):
        print("Model is inconsistent")
        return False
    else:
        r = np.dot(model - ground_truth, model - ground_truth) / (np.dot(ground_truth, ground_truth))
        print(r)
        if r < 0.0005:
            print(True)
            return True
        else:
            print(False)
            return False


# Ex1: make the same with polynoms

# generates x and y numpy arrays for
# y = a_n*X^n + ... + a2*x^2 + a1*x + a0 + noise
# in range -1 .. 1
# with random noise of given amplitude (noise)
# visualize it and unloads to csv
def generate_poly(a, n, noise, filename, size=100):
    x = 2 * np.random.rand(size, 1) - 1
    y = np.zeros((size, 1))
    print(np.shape(x))
    print(np.shape(y))
    if len(a) != (n + 1):
        print(f'ERROR: Length of polynomial coefficients ({len(a)}) must be the same as polynomial degree {n}')
        return
    for i in range(0, n + 1):
        y = y + a[i] * np.power(x, i) + noise * (np.random.rand(size, 1) - 0.5)
    data = np.hstack((x, y))
    np.savetxt(filename, data, delimiter=',')


def polynomial_regression_numpy(filename):
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)
    # our model
    time_start = time()
    model = np.polyfit(np.transpose(x)[0], np.transpose(y)[0], 2)
    time_end = time()
    print(f"polyfit in {time_end - time_start} seconds")
    # our hypothesis for give x
    print(model)
    # and check if it's ok
    plt.title("Polynomial regression task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    x = np.sort(x, axis=0)
    h = model[0] * np.power(x, 2) + model[1] * x + model[2]
    plt.plot(x, h, "r", label='model')
    plt.legend()
    plt.show()
    return model


# Ex.2 gradient descent for linear regression without regularization

# find minimum of function J(theta) using gradient descent
# alpha - speed of descend
# theta - vector of arguments, we're looking for the optimal ones (shape is 1 х N)
# J(theta) function which is being minimizing over theta (shape is 1 x 1 - scalar)
# dJ(theta) - gradient, i.e. partial derivatives of J over theta - dJ/dtheta_i
# (shape is 1 x N - the same as theta)
# x and y are both vectors


def gradient_descent_step(dJ, theta, alpha):
    theta = theta - alpha * dJ
    return theta


# get gradient over all xy dataset - gradient descent
def get_dJ(x, y, theta):
    h = np.dot(theta, x.transpose())  # calculate hypothesis
    dj = 1 / len(x) * (h - y.transpose()).dot(x)
    return dj


def minimize(theta, x, y, L):
    alpha = 0.1
    for i in range(L):
        dj = get_dJ(x, y, theta)  # here you should try different gradient descents
        theta = gradient_descent_step(dj, theta, alpha)
        h = np.dot(theta, x.transpose())
        j = 0.5 / len(x) * np.square(h - y.transpose()).sum(axis=1)
        alpha -= 0.0002
        plt.title("Minimization J")
        plt.xlabel("i")
        plt.ylabel("J")
        plt.plot(i, j, "b.")
    plt.show()
    return theta


# get gradient over all minibatch of size M of xy dataset - minibatch gradient descent
def minimize_minibatch(x, y, theta, l, m):
    alpha = 0.001
    for i in range(l):
        x = np.hstack([np.ones((800, 1)), x])
        x = np.vsplit(x, np.shape(x)[0] / m)
        y = np.vsplit(y, np.shape(y)[0] / m)
        for split_x, split_y in list(zip(x, y)):
            split_x.transpose()
            split_y.transpose()
            dj = get_dJ(split_x, split_y, theta)
            theta = gradient_descent_step(dj, theta, alpha)
            h = np.dot(theta, split_x.transpose())
            j = 0.5 / len(split_x) * np.square(h - split_y.transpose()).sum(axis=1)
            alpha += 0.0002
            plt.title("Minimization J for minibatch")
            plt.xlabel("i")
            plt.ylabel("J")
        plt.plot(i, j, "b.")
        generate_linear(1, -3, 1, 'linear_for_minibatch.csv', 800)
        with open('linear_for_minibatch.csv', 'r') as f:
            data = np.loadtxt(f, delimiter=',')
        x, y = np.hsplit(data, 2)
    plt.show()
    return theta


# get gradient over all minibatch of single sample from xy dataset - stochastic gradient descent
def minimize_sgd(x, y, theta, l):
    alpha = 0.1
    for i in range(l):
        #index = random.randint(0, len(x)-1)
        sgd_x = np.reshape(x, (1, 1))
        sgd_x = np.hstack([np.ones((1, 1)), sgd_x]).reshape(1, 2)
        sgd_y = np.reshape(y, (1, 1))
        dj = get_dJ(sgd_x, sgd_y, theta)
        theta = gradient_descent_step(dj, theta, alpha)
        h = np.dot(theta, sgd_x.transpose())
        j = 0.5 / len(x) * np.square(h - sgd_y.transpose()).sum(axis=1)
        alpha += 0.005
        plt.title("Minimization J for sgd")
        plt.xlabel("i")
        plt.ylabel("J")
        plt.plot(i, j, "b.")
        generate_linear(1, -3, 1, 'linear_for_sgd.csv', 1)
        with open('linear_for_sgd.csv', 'r') as f:
            data = np.loadtxt(f, delimiter=',')
        x, y = np.hsplit(data, 2)
    plt.show()
    return theta


def divide_minibatch(x, size):
    return x


if __name__ == "__main__":
    generate_linear(1, -3, 1, 'linear.csv', 100)
    model = linear_regression_numpy("linear.csv")
    print(f"Is model correct?\n{check(model, np.array([1, -3]))}")

    # ex1 . - exact solution
    model_exact = linear_regression_exact("linear.csv")
    check(model_exact, np.array([-3, 1]))

    # ex1. polynomial with numpy
    generate_poly([1, 2, 3], 2, 0.5, 'polynomial.csv')
    polynomial_regression_numpy("polynomial.csv")

    first = random.random()  # generate theta for minimize
    second = random.random()
    theta = np.array([first, second]).reshape((1, 2))

    with open('linear.csv', 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)
    x = np.hstack([np.ones((100, 1)), x])

    theta_grad = minimize(theta, x, y, 100)
    check(theta_grad[0], np.array([-3, 1]))

    generate_linear(1, -3, 1, 'linear_for_minibatch.csv', 800)
    with open('linear_for_minibatch.csv', 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)
    theta_grad = minimize_minibatch(x, y, theta, 80, 80)
    check(theta_grad[0], np.array([-3, 1]))

    generate_linear(1, -3, 1, 'linear_for_sgd.csv', 1)
    with open('linear_for_sgd.csv', 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)
    theta_grad = minimize_sgd(x, y, theta, 120)
    check(theta_grad[0], np.array([-3, 1]))
    # 3. call check(theta1, theta2) to check results for optimal theta

    # ex3. polinomial regression
    # 0. generate date with function generate_poly for degree=3, use size = 10, 20, 30, ... 100
    # for each size:
    # 1. shuffle data into train - test - valid
    # Now we're going to try different degrees of model to aproximate our data,
    # set degree=1 (linear regression)
    # 2. call minimize(...) and plot J(i)
    # 3. call check(theta1, theta2) to check results for optimal theta
    # 4. plot min(J_train), min(J_test) vs size: is it overfit or underfit?
    #
    # repeat 0-4 for degres = 2,3,4

    # ex3* the same with regularization
