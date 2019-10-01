import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sp
def display_data(x_mat):
    """displays 100 random images"""
    rand = np.random.randint(0,5000,(100,))
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.imshow(np.reshape(x_mat[rand[i],:],(20,20),order='F'),cmap='gray')
        plt.axis('off')
    plt.show()

def sigmoid(x_mat):
    """Calculates sigmoid function"""
    return 1/(1+np.exp(-x_mat))

def sigmoid_grad(x_mat):
    g = sigmoid(x_mat)
    return g*(1-g)

def indicator(y_mat, n_labels):
    """ calculates indicator matrix"""
    n_samples = y_mat.shape[0]
    y_indic = np.zeros((n_samples, n_labels))
    for i in range(n_samples):
        y_indic[i, y_mat[i]-1]=1
    return y_indic

def cost_function(x_mat, y_mat, theta1, theta2, n_labels, lamda):
    """Calculates cost and gradient"""
    y_indic = indicator(y_mat, 10)
    n_samples = x_mat.shape[0]
    a1 = np.append(np.ones((n_samples,1)), x_mat, axis=1)
    z2 = a1.dot(theta1.T)
    a2 = sigmoid(z2)
    a2 = np.append(np.ones((n_samples,1)), a2, axis = 1)
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)
    cost = 0
    for i in range(n_labels):
        cost = cost + -y_indic[:,i].T.dot(np.log(a3[:,i]))-(1-y_indic[:,i]).T.dot(np.log(1-a3[:,i]))
    cost = cost + lamda/2*((theta1[:,1:]**2).sum()+((theta2[:,1:])**2).sum())
    theta1_grad = 0
    theta2_grad = 0
    for i in range(n_samples):
        #Check the dimensions
        a1 = np.reshape(np.append(1, x_mat[i,:]),(1,401))
        z2 = a1.dot(theta1.T)
        a2 = sigmoid(z2)
        a2 = np.reshape(np.append(1, a2),(1,26))
        z3 = a2.dot(theta2.T)
        a3 = sigmoid(z3)
        delta3 = a3-y_indic[i,:]
        delta2 = (theta2.T.dot(delta3.T)).T*np.append(1, sigmoid_grad(z2))
        delta2 =  np.reshape(delta2[0,1:],(1,25))
        theta1_grad = theta1_grad+delta2.T.dot(a1)
        theta2_grad = theta2_grad+delta3.T.dot(a2)
    theta1_grad = theta1_grad/n_samples + lamda/n_samples*theta1
    theta2_grad = theta2_grad/n_samples + lamda/n_samples*theta2
    return cost/n_samples, theta1_grad, theta2_grad

def rand_initialize_weight(L_in, L_out):
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1+L_in)*2*epsilon_init-epsilon_init
    return W

def gradient_descent(x_mat, y_mat, theta1, theta2, n_labels, lamda, iter):
    """ applys gradient descent slgorithm"""
    print("Gradient descent algorithm running...")
    alpha = 2
    cost_history = np.zeros((iter,1))
    for i in range(iter):
        print(i)
        cost, theta1_grad, theta2_grad = cost_function(x_mat, y_mat, theta1, theta2, n_labels, lamda)
        cost_history[i,0] = cost
        theta1 = theta1 - alpha*theta1_grad
        theta2 = theta2 - alpha*theta2_grad
    print("cost is \n"+str(cost))
    return theta1, theta2, cost_history

def predict(x_mat, y_mat, theta1, theta2):
    """ Calculates the accuracy of the NN"""
    n_samples = x_mat.shape[0]
    a1 = np.append(np.ones((n_samples,1)), x_mat, axis=1)
    z2 = a1.dot(theta1.T)
    a2 = sigmoid(z2)
    a2 = np.append(np.ones((n_samples,1)), a2, axis = 1)
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)
    max_indx = np.reshape(np.argmax(a3, axis=1)+1,(n_samples,1))
    accuracy = 1*(y_mat == max_indx).sum()/n_samples*100
    return accuracy


def main():
    data = sp.loadmat('ex4data1')
    x_mat = data["X"]
    y_mat = data["y"]
    n_labels = 10
    #display_data(x_mat)
    weights = sp.loadmat("ex4weights")
    theta1 = weights["Theta1"]
    theta2 = weights["Theta2"]
    lamda = 1
    theta1_initial = rand_initialize_weight(400, 25)
    theta2_initial = rand_initialize_weight(25,10)
    iter = 300
    theta1, theta2, cost = gradient_descent(x_mat, y_mat, theta1_initial, theta2_initial, n_labels, lamda, iter)
    plt.plot(np.linspace(0,iter,iter),cost)
    plt.show()
    accuray = predict(x_mat, y_mat, theta1, theta2)
    print(accuray)


if __name__=='__main__':
    main()