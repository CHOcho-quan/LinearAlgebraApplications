import numpy as np

# np.set_printoptions(threshold=np.inf)

def buildTable(H):
    n_k, n = H.shape
    E = np.insert(np.identity(n), 0, values=np.zeros(n), axis=1)
    S = H.dot(E)
    return E, S

def buildTable2(H):
    n_k, n = H.shape
    E = []
    for i in range(n):
        for j in range(n):
            if i >= j:
                continue
            base = np.zeros(n)
            base[i] = 1
            base[j] = 1
            E.append(base)
    E = np.array(E).T
    # print("Eshape: ", E.shape)
    S = np.mod(H.dot(E), 2)
    return E, S

if __name__ == '__main__':
    test = np.concatenate((np.identity(5), np.zeros([5, 10])), axis=1)
    E, S = buildTable(test)
    print(E, E.shape)
    print(S, S.shape)