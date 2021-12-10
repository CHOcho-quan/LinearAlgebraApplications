import numpy as np
import buildTable as bT

def channelDecode(H, r):
    s = np.mod(H.dot(r), 2)
    E, S = bT.buildTable(H)
    ind = np.squeeze(np.argwhere([(S[:, i] == s).all() for i in range(S.shape[1])]))
    e = E[:, ind]
    x = np.abs(r - e)
    return x, e

def channelDecode2(H, r):
    s = np.mod(H.dot(r), 2)
    # print("Syndrome: ", s)
    E, S = bT.buildTable2(H)
    ind = np.squeeze(np.argwhere([(S[:, i] == s).all() for i in range(S.shape[1])]))
    # print(ind)
    ind = ind[0]
    e = E[:, ind]
    x = np.abs(r - e)
    return x, e

if __name__ == '__main__':
    H = np.array([[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
                  [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                  [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
                  [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1]])
    r = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    x, e = channelDecode(H, r)
    print(x)
    print(e)
