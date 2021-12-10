import numpy as np

def checkCodeword(H, x):
    res = np.dot(H, x)
    res = 1 - np.mod(res, 2)
    return res.all()

if __name__ == '__main__':
    H = np.array([[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
                  [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                  [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
                  [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1]])
    x1 = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    x2 = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    x3 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    print("x1 is", "in" if checkCodeword(H, x1) else "not in", "the null space of H" )
    print("x2 is", "in" if checkCodeword(H, x2) else "not in", "the null space of H" )
    print("x3 is", "in" if checkCodeword(H, x3) else "not in", "the null space of H" )
