import numpy as np
import pandas as pd
import scipy.io as sp
from matplotlib import pyplot as plt

def display_data(x_mat, y_mat, theta):
    '''visualizes the test data'''
    plt.scatter(x_mat, y_mat)
    #n_points = 200
    #x_vect = np.reshape(np.linspace(-50,40,num=n_points), (n_points,1))
    y_vect = np.append(np.ones((x_mat.shape[0],1)),x_mat, axis=1).dot(theta)
    plt.plot(x_mat, y_vect)
    plt.show()

def cost_function(x_mat, y_mat, theta, lamda):
    ''' calculates cost function'''
    grad = np.zeros(theta.shape)
    n_samples = x_mat.shape[0]
    cost = 1/(2*n_samples)*(x_mat.dot(theta)-y_mat).T.dot(x_mat.dot(theta)-y_mat)+lamda/(2*n_samples)*(theta[1:,:]**2).sum()
    grad = 1/n_samples*x_mat.T.dot(x_mat.dot(theta)-y_mat)
    grad[1:,0] = grad[1:,0] + lamda/n_samples*theta[1:,0]
    return cost, grad

def gradient_descent(x_mat, y_mat, theta, lamda, iter):
    '''gradient descent algorithm'''
    alpha = 0.05
    cost_history =  np.zeros(iter)
    n_samples = x_mat.shape[0]
    x_mat = np.append(np.ones((n_samples,1)), x_mat, axis = 1)
    for i in range(iter):
        cost, grad = cost_function(x_mat, y_mat, theta, lamda)
        cost_history[i] = cost
        theta = theta - alpha*grad
    return cost_history, theta

def normalization(x_mat):
    x_mean = np.mean(x_mat, axis=0)
    x_std = np.std(x_mat, axis=0)
    x_norm = (x_mat-x_mean)/x_std
    return x_norm, x_mean, x_std

def learning_curve(x_mat, y_mat, x_val, y_val, lamda, power):
    n_samples = x_mat.shape[0]
    iter = 40
    error_train = np.zeros(n_samples-1)
    error_val =  np.zeros(n_samples-1)
    theta = np.ones((x_mat.shape[1]+1,1))
    for i in range(2,n_samples+1):
        print(i)
        x_train = np.reshape(x_mat[0:i,:],(i,power),order='C')
        y_train = np.reshape(y_mat[0:i,:],(i,1),order='C')
        x_norm, x_mean, x_std = normalization(x_train)
        cost_history, theta = gradient_descent(x_norm, y_train, theta, lamda, iter)
        x_norm = np.append(np.ones((x_norm.shape[0],1)), x_norm, axis = 1)
        cost , grad = cost_function(x_norm, y_train, theta, 0)
        error_train[i-2] = cost
        x_val_norm = (x_val-x_mean)/x_std
        x_val_norm = np.append(np.ones((x_val_norm.shape[0],1)), x_val_norm, axis = 1)
        error_val[i-2], grad = cost_function(x_val_norm, y_val, theta, 0)
    plt.plot(np.linspace(2,n_samples,n_samples-1),error_train)
    plt.plot(np.linspace(2,n_samples,n_samples-1),error_val)
    plt.ylim(0,100)
    plt.show()

def poly_features(x_mat, power):
    ''' adds polynomial features'''
    x_poly = np.zeros((x_mat.shape[0], power))
    a = x_poly[:,0]
    b =  x_mat**(1)
    for i in range(power):
        x_poly[:,i] = np.reshape(x_mat**(i+1), (x_mat.shape[0],))
    return x_poly

def main():
    data = sp.loadmat('ex5data1.mat')
    x_mat = data["X"]
    y_mat = data["y"]
    x_test = data["Xtest"]
    y_test = data["ytest"]
    x_val = data["Xval"]
    y_val = data["yval"]
    lamda = 1
    theta = np.ones((x_mat.shape[1]+1,1))
    #x_mat = np.append(np.ones((x_mat.shape[0],1)), x_mat, axis = 1)
    #cost, grad = cost_function(x_mat, y_mat, theta, lamda)
    #print(cost)
    #print(grad)
    iter = 20
    #x_norm, x_mean, x_std = normalization(x_mat)
    #cost_history, theta = gradient_descent(x_norm, y_mat, theta, lamda, iter)
    #x_vect = np.linspace(1,iter,iter)
    #plt.plot(x_vect, cost_history)
    #plt.show()
    #isplay_data(x_norm, y_mat, theta)
    #learning_curve(x_mat, y_mat, x_val, y_val, lamda)
    #===========  Learning Curve for Polynomial Regression =============
    power = 8
    x_train_poly = poly_features(x_mat, power)
    x_norm, x_mean, x_std = normalization(x_train_poly)
    cost_history, theta = gradient_descent(x_norm, y_mat, np.ones((x_train_poly.shape[1]+1,1)), 0, iter)
    x_vect = np.linspace(1,iter,iter)
    plt.plot(x_vect, cost_history)
    plt.show()
    #print(x_train_poly[0:3,:])
    x_val_poly = poly_features(x_val, power)
    learning_curve(x_train_poly, y_mat, x_val_poly, y_val, 1, power)

if __name__=='__main__':
    main()
