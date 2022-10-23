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
    print('Generating random data y = a*x + b')
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
    dj = 1/len(x) * (h - y.transpose()).dot(x)
    return dj


# get gradient over all minibatch of size M of xy dataset - minibatch gradient descent
def get_dJ_minibatch(x, y, theta, M):
    new_theta = np.ndarray
    new_theta.reshape(np.shape(theta))
    theta_trans = theta.transpose()
    h = np.dot(theta_trans, x)  # calculate hypothesis
    delta = 0
    dj = [0] * len(theta)
    while M + delta < len(x):
        for i in range(len(theta)):  # calculate partial derivatives of J
            for k in range(len(x)):
                dj[i] += 1 / len(x) * (h[np.power(x, i)] - np.power(y, i)) * np.power(x[k], i)
            #new_theta.itemset(i, gradient_descent_step(dj[i], theta[i], alpha))
            delta = M + delta
    return new_theta


# get gradient over all minibatch of single sample from xy dataset - stochastic gradient descent
def get_dJ_sgd(x, y, theta, alpha):
    temp_x = x
    temp_y = y
    new_theta = np.ndarray
    new_theta.reshape(np.shape(theta))
    theta_trans = theta.transpose()
    h = np.dot(theta_trans, x)  # calculate hypothesis
    dj = [0] * len(theta)
    for i in range(len(theta)):  # calculate partial derivatives of J
        for k in range(len(x)):
            dj[i] += 1 / len(x) * (h[np.power(x, i)] - np.power(y, i)) * np.power(x[k], i)
    for i in range(len(x)):
        index = random.randint(0, len(x))
        new_theta.itemset(i, gradient_descent_step(dj[index], theta[index], alpha))
    return new_theta


def divide_x(x):
    random.shuffle(x)
    size_educational = 0.6 * len(x)
    size_test = 0.2 * len(x)  # also size of validate sample = size of test sample
    x_edu = list()
    x_t = list()
    x_v = list()
    for i in range(int(size_educational)):
        x_edu.append(x[i][0])
    for i in range(int(size_educational), int(size_educational) + int(size_test)):
        x_t.append(x[i][0])
    for i in range(int(size_educational) + int(size_test), int(len(x))):
        x_v.append(x[i][0])
    x_educational = np.array(x_edu).reshape(size_educational, 1)
    x_test = np.array(x_t).reshape(size_test, 1)
    x_validate = np.array(x_v).reshape(size_test, 1)
    return x_educational, x_validate, x_test


# try each of gradient decsent (complete, minibatch, sgd) for varius alphas
# L - number of iterations
# plot results as J(i)
def minimize(theta,x, y, L):
    # n - number of samples in learning subset, m - ...
    n = 100
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

    first = random.random()
    second = random.random()
    theta = np.array([first, second]).reshape((1, 2))

    with open('linear.csv', 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)
    x = np.hstack([np.ones((100, 1)), x])

    theta_grad = minimize(theta, x, y, 100)
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
