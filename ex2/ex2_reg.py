""" Coursera exercise 2: Regularized Logistic Regression """
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_data():
    """ Reads the data and converts to numpy array """
    data_frame = pd.read_csv("ex2data2.txt", header=None)
    data = data_frame.values
    n_sample = data.shape[0]
    x_mat = data[:, 0:2]
    y_mat = np.reshape(data[:, 2], (n_sample, 1))
    return x_mat, y_mat
def plot_data(x_mat, y_mat):
    """ Visualize the data, y=+, acceppted, y=o, rejected """
    y_mat = np.reshape(y_mat, (y_mat.shape[0],))
    x_1 = x_mat[:, 0]
    x_2 = x_mat[:, 1]
    x1_pass = x_1[y_mat > 0]
    x1_fail = x_1[y_mat < 1]
    x2_pass = x_2[y_mat > 0]
    x2_fail = x_2[y_mat < 1]
    plt.scatter(x1_pass, x2_pass, marker='+')
    plt.scatter(x1_fail, x2_fail, marker='o')
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    plt.show()
def map_feature(x_1,x_2,m):
    """ map the features into all polynomial terms of x1 and x2 up to the sixth power """
    x_mat_fm = np.zeros((m, 28))
    x_mat_fm[:, 0] = 1
    x_mat_fm[:, 1] = x_1
    x_mat_fm[:, 2] = x_2
    x_mat_fm[:, 3] = x_1**2
    x_mat_fm[:, 4] = x_1*x_2
    x_mat_fm[:, 5] = x_2**2
    x_mat_fm[:, 6] = x_1**3
    x_mat_fm[:, 7] = x_1**2*x_2
    x_mat_fm[:, 8] = x_1*x_2**2
    x_mat_fm[:, 9] = x_2**3
    x_mat_fm[:, 10] = x_1**4
    x_mat_fm[:, 11] = x_1**3*x_2
    x_mat_fm[:, 12] = x_1**2*x_2**2
    x_mat_fm[:, 13] = x_1*x_2**3
    x_mat_fm[:, 14] = x_2**4
    x_mat_fm[:, 15] = x_1**5
    x_mat_fm[:, 16] = x_1**4*x_2
    x_mat_fm[:, 17] = x_1**3*x_2**2
    x_mat_fm[:, 18] = x_1**2*x_2**3
    x_mat_fm[:, 19] = x_1*x_1**4
    x_mat_fm[:, 20] = x_2**5
    x_mat_fm[:, 21] = x_1**6
    x_mat_fm[:, 22] = x_1**5*x_2
    x_mat_fm[:, 23] = x_1**4*x_2**2
    x_mat_fm[:, 24] = x_1**3*x_2**3
    x_mat_fm[:, 25] = x_1**2*x_2**4
    x_mat_fm[:, 26] = x_1*x_2**5
    x_mat_fm[:, 27] = x_2**6
    return x_mat_fm
def cost_function_reg(x_mat, y_mat, lamda, theta):
    """ calculates the cost function and the gradient """
    grad = np.zeros(theta.shape)
    n_samples = x_mat.shape[0]
    estimate = sigmoid(x_mat.dot(theta))
    cost = 1/n_samples*(-y_mat.T.dot(np.log(estimate))-(1-y_mat).T.dot(np.log(1-estimate))) \
    +lamda/(2*n_samples)*(theta[1:, :]**2).sum()
    grad[0, 0] = 1/n_samples*(estimate-y_mat).sum()
    grad[1:, :] = 1/n_samples*x_mat[:, 1:].T.dot(estimate-y_mat)+lamda/n_samples*theta[1:, :]
    return cost, grad
def sigmoid(x_mat):
    """ Calculates sigoid function"""
    return 1/(1+np.exp(-x_mat))
def normalize(x_mat):
    """ Normalization function"""
    y_mat = np.zeros(x_mat.shape)
    y_mat[:,0]=1
    x_mean = np.mean(x_mat[:,1:], axis=0)
    x_std = np.std(x_mat[:,1:], axis=0)
    y_mat[:,1:] = (x_mat[:,1:]-x_mean)/x_std
    return y_mat, x_mean, x_std
def gradient_descent(x_mat, y_mat, theta, alpha, lamda, iter):
    """ Calculates optimum theta using gradient descent algorithm"""
    x_mat_n, x_mean, x_std = normalize(x_mat)
    cost_history = np.zeros(iter)
    for i in range(iter):
        cost, grad = cost_function_reg(x_mat_n, y_mat, lamda, theta)
        theta = theta - alpha*grad
        cost_history[i] = cost
    theta_denorm = np.zeros(theta.shape)
    theta_denorm[0,0] = theta[0,0] - (theta[1:, 0]*x_mean/x_std).sum()
    theta_denorm[1:, 0] = theta[1:, 0]/x_std
    return theta_denorm, cost_history
def plot_decision_boundary(x_mat, y_mat, theta):
    """ Plot the decision boundary"""
    y_mat = np.reshape(y_mat, (y_mat.shape[0],))
    x_1 = x_mat[:, 0]
    x_2 = x_mat[:, 1]
    x1_pass = x_1[y_mat > 0]
    x1_fail = x_1[y_mat < 1]
    x2_pass = x_2[y_mat > 0]
    x2_fail = x_2[y_mat < 1]
    plt.scatter(x1_pass, x2_pass, marker='+')
    plt.scatter(x1_fail, x2_fail, marker='o')
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    u = np.linspace(-1,1.5)
    v = np.linspace(-1,1.5)
    z = np.zeros((u.shape[0],v.shape[0]))
    for i in range(u.shape[0]):
        for j in range(v.shape[0]):
            z[i,j] = map_feature(u[i],v[j],1).dot(theta)
    plt.contour(u,v,z, levels=0)
    plt.show()
def main():
    """ main function"""
    x_mat, y_mat = load_data()
    plot_data(x_mat, y_mat)
    x_1 = x_mat[:, 0]
    x_2 = x_mat[:, 1]
    x_mat_fm = map_feature(x_1, x_2, x_1.shape[0])
    initial_theta = np.zeros((x_mat_fm.shape[1], 1))
    cost, grad = cost_function_reg(x_mat_fm, y_mat, 1, initial_theta)
    print("cost at lambda=1 \n")
    print(cost)
    print("first five gradients: \n")
    print(grad[0:5])
    test_theta = np.ones((x_mat_fm.shape[1], 1))
    cost, grad = cost_function_reg(x_mat_fm, y_mat, 10, test_theta)
    print("cost at lambda=10 \n"+ str(cost))
    print("first five gradients: \n")
    print(grad[0:5])
    iter = 10000
    alpha = 1
    lamda = 1
    theta, cost_history = gradient_descent(x_mat_fm, y_mat, initial_theta, alpha, lamda, iter)
    plt.plot(np.linspace(1,iter,iter),cost_history)
    plt.show()
    #print(theta)
    plot_decision_boundary(x_mat, y_mat, theta)
if __name__ == '__main__':
    main()
