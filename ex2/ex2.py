import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plotData(X,Y):
    """ Function to plot the data """
    X1 = X[:,0]
    X2 = X[:,1]
    X1_pass = X1[Y>0]
    X2_pass = X2[Y>0]
    X1_fail = X1[Y<1]
    X2_fail = X2[Y<1]
    plt.scatter(X1_pass,X2_pass, marker='+',label='Admitted')
    plt.scatter(X1_fail,X2_fail,marker='o',label = 'Not Admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    plt.show()

def feature_normalize(x):
    x_mean = np.mean(x,axis=0)
    x_std = np.std(x,axis=0)
    return (x-x_mean)/x_std,x_mean,x_std

def sigmoid(x):
    return(1/(1+np.exp(-x)))

def costFunction(X,Y,theta):
    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    J = 1/m*(-Y.T.dot(np.log(h))-(1-Y).T.dot(np.log(1-h)))
    grad = 1/m*(X.T.dot(h-Y))
    return J,grad

def GradientDescend(X,Y,theta,num_iter,learning_rate):
    cost_history = np.zeros((num_iter,1))
    for i in range(num_iter):
        J,grad = costFunction(X,Y,theta)
        theta = theta -learning_rate*grad
        cost_history[i,0]=J
    return theta, cost_history

def plotDecisionBoundary(X,Y,theta):
    X1 = X[:,0]
    X2 = X[:,1]
    X1_pass = X1[Y>0]
    X2_pass = X2[Y>0]
    X1_fail = X1[Y<1]
    X2_fail = X2[Y<1]
    plt.scatter(X1_pass,X2_pass, marker='+',label='Admitted')
    plt.scatter(X1_fail,X2_fail,marker='o',label = 'Not Admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    Boundary=(-theta[0,0]-theta[1,0]*X1)/theta[2,0]
    plt.plot(X1,Boundary)
    plt.show()

#def prdict()
#def costFunctionReg()
## ==================== Part 1: Plotting ====================
print('Loading data ...\n')
# Load data
data = pd.read_csv('ex2data1.txt',header=None) 
M = data.values
X = M[:,0:2]
shape = X.shape
m = shape[0]
n = shape[1]
Y = np.reshape(M[:,2],(m,1))

# Plot the data
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
plotData(X,M[:,2])

input('\nProgram paused. Press enter to continue.\n')
# Normalization
X,X_mean,X_std = feature_normalize(X)
## ============ Part 2: Compute Cost and Gradient ============
X = np.append(np.ones((m,1)),X,axis=1)
theta_init = np.zeros((n+1,1))
#theta_init = np.array([[-24],[0.2],[0.2]])
J,grad = costFunction(X,Y,theta_init)
#print(J)
#print(grad)
learning_rate = 1
num_iter = 2000
theta, cost_history = GradientDescend(X,Y,theta_init,num_iter,learning_rate)
plt.plot(np.linspace(1,num_iter,num_iter),cost_history)
plt.show()
theta_den = np.zeros((n+1,1))
theta_den[0,0] = theta[0,0]-theta[1,0]*X_mean[0]/X_std[0]-theta[2,0]*X_mean[1]/X_std[1]
theta_den[1,0] = theta[1,0]/X_std[0]
theta_den[2,0] = theta[2,0]/X_std[1]
#print(theta)
print(theta_den)
#print(cost_history[num_iter-1])

plotDecisionBoundary(M[:,0:2],M[:,2],theta_den)

