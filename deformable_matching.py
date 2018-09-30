from functools import partial
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from deformable_registration import *

import numpy as np
import time

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:,0] ,  X[:,1], color='red')
    ax.scatter(Y[:,0] ,  Y[:,1], color='blue')


    min = [177, 157, 199, 206,  21, 111, 128, 129,  92,  95]
    # ax.scatter(Y[128, 0], Y[128, 1], color='green')
    plt.draw()
    print("iteration %d, error %.5f" % (iteration, error))
    plt.pause(0.1)

def main():
    fish = loadmat('two_data_sets.mat')
    # fish = loadmat('fish.mat')
    X = fish['X']
    Y = fish['Y']
    print(X.shape, Y.shape)
    # test synthetic data
    xx = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6.0, 0],
                   [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6.0, 1],
                   [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6.0, 2],
                   [1, 3], [2, 3], [3, 3], [4, 3], [5, 3], [6.0, 3]])
    yy = np.array([[1, 0.1], [2, 0.4], [3, 0.6], [4, 0.5], [5, 0.3], [6, 0.1],
                   [1, 1.1], [2, 1.3], [3, 1.6], [4, 1.5], [5, 1.3], [6.0, 1.1],
                   [1, 2.2], [2, 2.4], [3, 2.6], [4, 2.4], [5, 2.3], [6.0, 2.1],
                   [1, 3.1], [2, 3.3],
                   [3, 3.6],
                   [4, 3.5], [5, 3.4], [6.0, 3.1]])

    savemat('synthetic_data_missing', mdict={'X': xx, 'Y': yy})

    print(xx.shape)
    # print(Y)
    #
    # plt.figure(1)
    # # plt.scatter(X[:,0], X[:,1])
    # plt.scatter(Y[:,0], Y[:,1])


    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])
    # reg = deformable_registration(xx, yy, tolerance=0.000001, sigma2=100)

    reg = deformable_registration(xx[:200, :], yy[:200, :], tolerance=0.000001, sigma2=100)

    reg.register(callback)
    plt.show()

if __name__ == '__main__':
    main()
