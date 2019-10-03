import numpy as np
import scipy.io as sp
from matplotlib import pyplot as plt
def find_closest_centroid(x_mat , centroids):
    n_samples = x_mat.shape[0]
    n_cluster = centroids.shape[0]
    idx = np.zeros((n_samples,1))
    dist = np.zeros((n_samples, n_cluster))
    for i in range(n_cluster):
        dist[:, i] = np.sum((x_mat - np.reshape(centroids[i, :],(1, 2)))**2, axis=1)
    idx = np.argmin(dist, axis=1)
    return idx

def main():
    data = sp.loadmat('ex7data2.mat')
    x_mat = data["X"]
    plt.scatter(x_mat[:, 0], x_mat[:, 1])
    #plt.show()
    # select an initial set of centroids
    n_cluster = 3
    initial_centroids =np.array([[3, 3], [6, 2], [8, 5]])
    idx = find_closest_centroid(x_mat , initial_centroids)
    print(idx[0:3])
if __name__ == "__main__":
    main()
