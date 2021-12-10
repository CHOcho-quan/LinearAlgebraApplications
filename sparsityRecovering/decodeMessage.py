import omp
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

if __name__ == '__main__':
    m = loadmat("./Y1Y2Y3andA1A2A3.mat")
    A1 = m["A1"]
    A2 = m["A2"]
    A3 = m["A3"]
    y1 = m["y1"] # 960 x 1
    y2 = m["y2"] # 1440 x 1
    y3 = m["y3"] # 2880 x 1

    # Display y1 y2 y3
    plt.imshow(y1.reshape((24, 40)))
    plt.imshow(y2.reshape((36, 40)))
    plt.imshow(y3.reshape((40, 72)))
    plt.show()

    # Recover X1 X2 X3 by OMP
    ak, sup = omp.OMP(A1, y1, -1, True, 0.001)
    x1_hat = np.zeros((14400, 1))
    for j in range(len(sup)):
        x1_hat[sup[j]] = ak[j]
    plt.imshow(np.flip(x1_hat.reshape(160, 90), 0))
    plt.colorbar()
    plt.show()

    ak, sup = omp.OMP(A2, y2, -1, True, 0.001)
    x2_hat = np.zeros((14400, 1))
    for j in range(len(sup)):
        x2_hat[sup[j]] = ak[j]
    plt.imshow(np.flip(x2_hat.reshape(160, 90), 0))
    plt.colorbar()
    plt.show()

    ak, sup = omp.OMP(A3, y3, -1, True, 0.001)
    x3_hat = np.zeros((14400, 1))
    for j in range(len(sup)):
        x3_hat[sup[j]] = ak[j]
    plt.imshow(np.flip(x3_hat.reshape(160, 90), 0))
    plt.colorbar()
    plt.show()

    # Recover X1 X2 X3 by Least Square
    x1_hat_ls = np.linalg.pinv(A1).dot(y1)
    plt.imshow(np.flip(x1_hat_ls.reshape(160, 90), 0))
    plt.colorbar()
    plt.show()

    x2_hat_ls = np.linalg.pinv(A2).dot(y2)
    plt.imshow(np.flip(x2_hat_ls.reshape(160, 90), 0))
    plt.colorbar()
    plt.show()

    x3_hat_ls = np.linalg.pinv(A3).dot(y3)
    plt.imshow(np.flip(x3_hat_ls.reshape(160, 90), 0))
    plt.colorbar()
    plt.show()
    