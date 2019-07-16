""" Coursera exercise 3 | Part 2: Neural Network """
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sp
def predict(x_mat, y_mat, theta1, theta2):
    """ Feedforward propagation"""
    n_samples = x_mat.shape[0]
    x_mat = np.append(np.ones((n_samples,1)),x_mat, axis=1)
    # z_1 has size 5000*25
    z_2 = x_mat.dot((theta1).T)
    a_2 = sigmoid(z_2)
    # a_2 has size 5000*26
    a_2 = np.append(np.ones((n_samples,1)), a_2, axis=1)
    #z_3 has size 5000*10
    z_3 = a_2.dot((theta2.T))
    a_3 = sigmoid(z_3)
    prediction = np.reshape(np.argmax(a_3, axis=1), (n_samples,1))+1
    precision = (1*(prediction==y_mat)).sum()/n_samples*100
    return precision, prediction

def sigmoid(x_mat):
    """Calculates sigmoid function"""
    return 1/(1+np.exp(-x_mat))

def show_image(x):
    """ Displays the image"""
    x = np.reshape(x,(20,20), order = 'F')
    plt.imshow(x,cmap='gray')
    plt.show()

def main():
    """ Main function"""
    # Load data set and NN weights
    dataset = sp.loadmat("ex3data1.mat")
    nn_weights = sp.loadmat("ex3weights.mat")
    # x_mat has size 5000*400
    x_mat = dataset["X"]
    # y_mat has size 5000*1
    y_mat = dataset["y"]
    # theta1 has size 25*401
    theta1 = nn_weights["Theta1"]
    #theta2 has size 10*26
    theta2 = nn_weights["Theta2"]
    n_samples = x_mat.shape[0]
    #n_features = x_mat.shape[1]
    #n_hidden = theta1.shape[1]
    #k_class = theta2.shape[1]
    precision, prediction = predict(x_mat, y_mat, theta1, theta2)
    print("The accuracy is about "+str(precision)+"%")
    randperm = np.arange(n_samples)
    np.random.shuffle(randperm)
    print(type(randperm))
    print(randperm.shape)
    for i in range(n_samples):
        x_sample = x_mat[randperm[i],:]
        x_sample = np.reshape(x_sample,(1,400))
        print('\nDisplaying Example Image\n')
        show_image(x_sample)
        precision, prediction= predict(x_sample, y_mat, theta1, theta2)
        if prediction==10:
            prediction=0
        print('\nNeural Network Prediction:'+str(prediction))
        input('Paused - press enter to continue')



if __name__=='__main__':
    main()