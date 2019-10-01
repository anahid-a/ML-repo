import numpy as np
import scipy.io as sp
from matplotlib import pyplot as plt
from sklearn import svm 
def classifier1():
    data = sp.loadmat('ex6data1')
    x_mat = data["X"]
    y_mat = data["y"]
    n_samples = y_mat.shape[0]
    pos = np.where(y_mat==1)
    neg = np.where(y_mat==0)
    x_pos = x_mat[pos[0], :]
    x_neg = x_mat[neg[0], :]
    model = svm.SVC(C = 100, kernel='linear')
    model.fit(x_mat, np.reshape(y_mat, (n_samples,)))
    # create a mesh to plot in
    h = .02
    x_min, x_max = x_mat[:, 0].min() - 1, x_mat[:, 0].max() + 1
    y_min, y_max = x_mat[:, 1].min() - 1, x_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(x_pos[:,0], x_pos[:,1], marker='+')
    plt.scatter(x_neg[:,0], x_neg[:,1], marker='o')
    plt.show()

def classifier2():
    data = sp.loadmat('ex6data2')
    x_mat = data["X"]
    y_mat = data["y"]
    n_samples = y_mat.shape[0]
    pos = np.where(y_mat==1)
    neg = np.where(y_mat==0)
    x_pos = x_mat[pos[0], :]
    x_neg = x_mat[neg[0], :]
    model = svm.SVC(C = 1, kernel='rbf', gamma=50)
    model.fit(x_mat, np.reshape(y_mat, (n_samples,)))
    h = .02
    x_min, x_max = x_mat[:, 0].min() - 0.1, x_mat[:, 0].max() + 0.1
    y_min, y_max = x_mat[:, 1].min() - 0.1, x_mat[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(x_pos[:,0], x_pos[:,1], marker='+')
    plt.scatter(x_neg[:,0], x_neg[:,1], marker='o')
    plt.show()

def classifier3():
    data = sp.loadmat('ex6data3')
    x_mat = data["X"]
    y_mat = data["y"]
    x_val = data["Xval"]
    y_val = data["yval"]
    n_samples = y_mat.shape[0]
    pos = np.where(y_mat==1)
    neg = np.where(y_mat==0)
    x_pos = x_mat[pos[0], :]
    x_neg = x_mat[neg[0], :]

    C, sigma = data3params(x_mat, y_mat, x_val, y_val)
    print(C)
    print(sigma)
    model = svm.SVC(C = C, kernel='rbf', gamma=1/(2*sigma**2))
    model.fit(x_mat, np.reshape(y_mat, (n_samples,)))

    h = .02
    x_min, x_max = x_mat[:, 0].min() - 0.1, x_mat[:, 0].max() + 0.1
    y_min, y_max = x_mat[:, 1].min() - 0.1, x_mat[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(x_pos[:,0], x_pos[:,1], marker='+')
    plt.scatter(x_neg[:,0], x_neg[:,1], marker='o')
    plt.show()

def data3params(x_mat, y_mat, x_val, y_val):
    n_samples = y_mat.shape[0]
    c_vect = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    sigma_vect = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    err = np.zeros((c_vect.shape[0], sigma_vect.shape[0]))
    for i in range(c_vect.shape[0]):
        for j in range(sigma_vect.shape[0]):
            model = svm.SVC(C = c_vect[i], kernel='rbf', gamma=sigma_vect[j])
            model.fit(x_mat, np.reshape(y_mat, (n_samples,)))
            Z = model.predict(x_val)
            err[i,j] = np.mean(1*(Z!=y_val))
    print(err)
    argmin_idx = np.unravel_index(err.argmin(), err.shape)
    return c_vect[argmin_idx[0]], sigma_vect[argmin_idx[1]]
def main():
    #classifier1()
    #classifier2()
    classifier3()

if __name__=='__main__':
    main()