import numpy as np
import channelDecode as cD

# np.set_printoptions(threshold=np.inf)

def possibleSignals(x):
    n = x.shape[0]
    error = np.identity(n)
    return np.mod(x + error, 2)

def possibleSignals2(x):
    n = x.shape[0]
    error = []
    for i in range(n):
        for j in range(n):
            if i >= j:
                continue
            base = np.zeros(n)
            base[i] = 1
            base[j] = 1
            error.append(base)
    
    error = np.array(error)
    return np.mod(x + error, 2)

if __name__ == '__main__':
    H = np.array([[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
                  [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                  [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
                  [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1]])
    x = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    print("All the possible signals are the row vectors of following matrix:")
    signals = possibleSignals2(x)
    print(signals)

    cnt = 0
    correct = 0
    for i in range(signals.shape[0]):
        cnt += 1
        signal = signals[i, :]
        # print(signal)
        decoded, _ = cD.channelDecode2(H, signal)
        # print("Decoded: ", decoded)
        if (decoded == x).all():
            correct += 1
    print("Correct Rate is:", float(correct) / cnt)
