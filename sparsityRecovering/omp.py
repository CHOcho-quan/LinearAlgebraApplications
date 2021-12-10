import random
import matplotlib.pyplot as plt
import numpy as np

def OMP(A, y, sparsity = -1, stop_by_criterion = False, criterion = 0.001, max_iter = 100000):
    """
    Main OMP Algorithm
    INPUT
    A - Matrix of signal generation
    y - generated signal
    sparsity - current sparsity level
    criterion - stopping criterion of OMP Algorithm
    max_iter - max iteration if dead while
    OUTPUT
    x - recovered signal
    Sigma - support of the signal
    
    """
    rk = y.copy()
    ak = np.zeros_like(rk)
    Sigma = np.array([], dtype=int)
    count = 0
    while (not stop_by_criterion and count < sparsity) or (stop_by_criterion and np.linalg.norm(rk) > criterion and count < max_iter):
        inner = np.squeeze(np.abs(np.inner(A.transpose(), rk.transpose())))
        lambdak = np.argmax(inner)
        Sigma = np.append(Sigma, lambdak)
        # Construct Q to find the projection of y to span(Sigma)
        Q = A[:, Sigma]
        ak = np.linalg.pinv(Q).dot(y)
        rk = y - Q.dot(ak)
        count = count + 1
    return ak, Sigma

def OMPLoop(N, M, s, l, sigma = 1.0, noiseless = True):
    """
    Loop to do OMP Algorithm for different cases
    INPUT
    N - column number of matrix A
    M - row number of matrix A, M < N
    s - current cardinality of S
    l - loop times for the OMP algorithm
    sigma - noise gaussian standard deviation
    noiseless - if add noise to the signal
    OUTPUT
    p - OMP Result performance

    """
    success = 0
    error = 0.0
    for i in range(l):
        A = np.random.normal(0, 1, (M, N))
        A = A / np.linalg.norm(A, None, 0)
        x = np.zeros((N, 1))
        noise = np.zeros((M, 1)) if noiseless else np.random.normal(0, sigma, (M, 1))
        # Put in s non-zero entries to x
        support = random.sample(range(1, N), s)
        for j in range(s):
            rand1 = random.uniform(-10, -1)
            rand2 = random.uniform(1, 10)
            x[support[j], 0] = rand1 if random.randint(1, 2) % 2 else rand2
        
        # Start OMP Algorithm
        y = A.dot(x) + noise
        res, res_sup = OMP(A, y, s, stop_by_criterion=True, criterion=np.linalg.norm(noise)) # OMP(A, y, s)

        # Recover X since columns of res_sups are linear independent
        x_hat = np.zeros_like(x)
        for j in range(len(res_sup)):
            x_hat[res_sup[j]] = res[j]

        # Calculate Errors
        cur_err = np.linalg.norm(x - x_hat) / np.linalg.norm(x)
        error = error + cur_err
        if noiseless:
            success = success + np.array_equal(np.sort(support), np.sort(res_sup))
        else:
            success = success + (cur_err < 10e-3)

    recover = float(success) / l
    norm_err = float(error) / l
    print("Exact Support Recovery Rate for N =", N, "M =", M, "S =", s, "is", recover)
    print("Average Normalized Error for N =", N, "M =", M, "S =", s, "is", norm_err)
    return recover, norm_err

if __name__ == '__main__':
    # Noiseless
    N = 100 # 50 100
    # recover_img = np.zeros((N - 1, N - 1))
    # normerr_img = np.zeros((N - 1, N - 1))
    # for M in range(1, N):
    #     for s in range(1, M + 1):
    #         recover, norm_err = OMPLoop(N, M, s, 200)
    #         recover_img[M - 1][s - 1] = recover
    #         normerr_img[M - 1][s - 1] = norm_err
    # fig1 = plt.figure()
    # plt.imshow(recover_img)
    # plt.colorbar()
    # plt.title("Recover Rate for Different M & s")
    # plt.xlabel("s")
    # plt.ylabel("M")
    # fig2 = plt.figure()
    # plt.imshow(normerr_img)
    # plt.colorbar()
    # plt.title("Norm Error for Different M & s")
    # plt.xlabel("s")
    # plt.ylabel("M")
    # plt.show()
    # With Noise
    recover_noise_img = np.zeros((N - 1, N - 1))
    normerr_noise_img = np.zeros((N - 1, N - 1))
    for M in range(1, N):
        for s in range(1, M + 1):
            recover, norm_err = OMPLoop(N, M, s, 200, sigma=0.1, noiseless=False)
            recover_noise_img[M - 1][s - 1] = recover
            normerr_noise_img[M - 1][s - 1] = norm_err
    fig3 = plt.figure()
    plt.imshow(recover_noise_img)
    plt.colorbar()
    plt.title("Recover Rate for Different M & s With Noise")
    plt.xlabel("s")
    plt.ylabel("M")
    fig4 = plt.figure()
    plt.imshow(normerr_noise_img)
    plt.colorbar()
    plt.title("Norm Error for Different M & s With Noise")
    plt.xlabel("s")
    plt.ylabel("M")
    plt.show()
