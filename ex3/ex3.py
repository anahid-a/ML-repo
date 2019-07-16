""" Coursera exercise 3 | Part 1: One vs All"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#from matplotlib import image as mpimg
import scipy.io as sp


def display_data(x_mat):
    """ Displays 100 random images"""
    n_max = 100
    row_rand = np.random.randint(0,x_mat.shape[0]-1,n_max)
    for i in range(n_max):
        img = np.reshape(x_mat[row_rand[i],:],(20,20),order='F')
        plt.subplot(np.sqrt(n_max),np.sqrt(n_max),i+1)
        plt.imshow(img,cmap='gray')
        plt.axis('off')
    plt.show()

def sigmoid(x_mat):
    """Sigmoid function"""
    return 1/(1+np.exp(-x_mat))

def indicator(y_mat, k_class):
    """indicator function of the outputs"""
    n_samples = y_mat.shape[0]
    y_indicator = np.zeros((n_samples,k_class))
    for i in range(n_samples):
        y_indicator[i,y_mat[i]]=1
    return y_indicator

def cost_function(x_mat, y_mat, theta_mat, lamda):
    """ Calculates cost and gradients"""
    grad = np.zeros(theta_mat.shape)
    n_samples = x_mat.shape[0]
    estimate = sigmoid(x_mat.dot(theta_mat))
    cost = 1/n_samples*(-y_mat.T.dot(np.log(estimate))-(1-y_mat).T.dot(np.log(1-estimate))) \
        + lamda/(2*n_samples)*(theta_mat[1:,:]**2).sum()
    grad[0,:] = 1/n_samples*(estimate-y_mat).sum()
    grad[1:,:] = 1/n_samples*x_mat[:,1:].T.dot(estimate-y_mat) + lamda/n_samples*theta_mat[1:,:]
    return cost, grad

def gradient_descent(x_mat, y_mat, theta, lamda):
    """ applys gradient descent slgorithm"""
    print("Gradient descent algorithm running...")
    alpha = 1
    iter = 5000
    cost_history = np.zeros((iter,1))
    for i in range(iter):
        cost, grad = cost_function(x_mat, y_mat, theta, lamda)
        cost_history[i,0] = cost
        theta = theta - alpha*grad
    print("cost is \n"+str(cost))
    return theta, cost_history

def one_vs_all(x_mat, y_mat, lamda):
    """ Performs logistic regression for each class: one versus all"""
    print("One vs all...")
    k_class = 10
    theta_mat = np.zeros((x_mat.shape[1], k_class))
    for k in range(k_class):
        print(str(k)+"\n")
        theta = np.zeros((x_mat.shape[1],1))
        y_indicator = 1*(y_mat==k)
        theta, cost_history = gradient_descent(x_mat, y_indicator, theta, lamda)
        theta_mat[:,k] = theta[:,0]
    return theta_mat

def prediction(x_mat, y_mat, theta_mat):
    """ Calculates the prediction error"""
    print("calculating predictions...")
    n_samples = x_mat.shape[0]
    estimate = sigmoid(x_mat.dot(theta_mat))
    predict = np.reshape(np.argmax(estimate, axis=1),(n_samples,1))
    precision = (1*(predict==y_mat)).sum()/n_samples*100
    return precision
def main():
    """ The main function"""
    # Load data
    mat = sp.loadmat('ex3data1.mat')
    # mat is a dictionary with keys: X and y
    x_mat = mat["X"]
    y_mat = mat["y"]
    y_mat[y_mat==10]=0
    k_class = 10
    n_samples = x_mat.shape[0]
    display_data(x_mat)
    x_mat = np.append(np.ones((n_samples,1)), x_mat, axis=1)
    theta_mat = np.zeros((x_mat.shape[1], k_class))
    # test cost function
    # print("Testing cost function...\n")
    # theta_t = np.array([[-2],[-1],[1],[2]])
    # X_t = np.append(np.ones((5,1)),np.reshape((np.array(range(15))+1)/10,(5,3), order='F'),axis=1)
    # Y_t = np.array([[1],[0],[1],[0],[1]])
    # lamda_t = 3
    # print(X_t)
    # print(Y_t)
    # print(theta_t)
    # cost, grad = cost_function(X_t, Y_t, theta_t, lamda_t)
    # print("cost")
    # print(cost)
    # print("gradient")
    # print(grad)
    lamda = 0.1
    theta_mat=one_vs_all(x_mat, y_mat, lamda)
    precision = prediction(x_mat, y_mat, theta_mat)
    print("The accuracy is:")
    print(str(precision) + "%")

if __name__=='__main__':
    main()

