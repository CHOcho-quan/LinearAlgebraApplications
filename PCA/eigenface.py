import cv2
import glob, random
import matplotlib.pyplot as plt
import numpy as np

def CalculateMSE(reconstructed, original):
    """
    Calculate MSE for reconstructed picture
    Input:
    reconstructed - reconstructed face image
    original - original face image
    Output:
    mse - min square error of the reconstructed graph

    """
    original = original.T
    mse = np.mean((reconstructed - original)**2, axis=0)
    return mse

def ReconstructFace(eig_face, img, num_PC):
    """
    Reconstruct Face Image by Calculated Eigen Faces
    Input:
    eig_face - eigen face vector
    img - image to be reconstruted
    num_PC - number of principal components to be used
    Output:
    reconstructed - reconstructed picture of the face
    
    """
    pc = eig_face[0:num_PC + 1, :]
    reconstructed = np.dot(pc.T, np.dot(pc, img.T))
    print("MSE using {0} Principal Components is".format(num_PC), CalculateMSE(reconstructed, img))
    return reconstructed

def ReadDataset(label, root_path="./eigenface"):
    """
    Read eigenface dataset according to input
    Input:
    label - string to choose smile or neutral faces
    root_path - root path of the dataset
    Output:
    datasets - list with all selected label pictures in it

    """
    files = glob.glob(root_path + "/" + "*{0}.jpg".format(label))
    files.sort()
    datasets = []
    for file in files:
        # 193 * 162 Image shape
        cur_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        datasets.append(cur_img.reshape(1, -1))
    
    return datasets

def CalculateEigenface(dataset, nums=0, label="neutral"):
    """
    Calculate the eigenface space of the given dataset
    Input:
    dataset - the given dataset to do eigenfaces
    nums - select nums meaningful component of eigenfaces
    label - input neutral or smile dataset
    Output:
    mean_face - mean face of the given dataset
    eig_face - calculated eigenface basis of the dataset

    """
    # First calculate the mean face
    if len(dataset) == 0:
        return None
    M = len(dataset)
    cur_data = np.squeeze(np.array(dataset))
    mean_face = np.mean(cur_data, axis=0).reshape(1, -1)
    # plt.imsave("./images/mean_face_{0}.jpg".format(label), mean_face.reshape(193, 162), cmap='gray')
    # plt.imshow(mean_face.reshape(193, 162), cmap='gray')
    # plt.show()

    # Get the covariance matrix of the faces
    # Simplified version in Turk's paper
    A = cur_data - mean_face
    cov_face = np.dot(A, A.T)

    # Calculate Eigen face Space
    eig_val, eig_vec = np.linalg.eig(cov_face)
    idx = eig_val.argsort()[::-1] 
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:,idx]

    # Show nums most representative eigenfaces
    for i in range(nums):
        plt.imshow(eig_vec[i].reshape(193, 162))
        plt.show()

    # Plot its singular values
    plt.stem(np.sqrt(eig_val), use_line_collection=True)
    my_x_ticks = np.arange(0, 100, 10)
    plt.xticks(my_x_ticks)
    # plt.savefig("./images/singular_value_{0}.jpg".format(label))
    # plt.show()

    eig_face = np.dot(A.T, eig_vec)
    eig_face = eig_face / np.linalg.norm(eig_face, axis = 0)

    return mean_face, eig_face.T

if __name__ == '__main__':
    dataset = ReadDataset('a') # Change to b for smile faces
    # WLOG Choose first 100 as training set
    train_data = dataset[:100]
    test_data = dataset[100:]
    mean_face, eig_face = CalculateEigenface(train_data, 100)

    # Reconstruct one of the training set image
    rand_img = train_data[random.randint(0, 99)] - mean_face

    mse_img = []
    for i in range(100):
        recon = ReconstructFace(eig_face, rand_img, i + 1)
        if (i == 0 or i == 49 or i == 99):
            plt.subplot(121)
            plt.imshow(rand_img.reshape(193, 162), cmap='gray')
            plt.subplot(122)
            plt.imshow(recon.reshape(193, 162), cmap='gray')
            plt.show()
        mse = CalculateMSE(recon, rand_img)
        mse_img.append(mse)
    plt.plot(list(range(100)), mse_img)
    plt.xlabel("PC Num")
    plt.ylabel("Reconstruction MSE")
    plt.show()

    # Reconstruct one of the testing set image
    rand_img = test_data[random.randint(0, 71)] - mean_face

    mse_img = []
    for i in range(100):
        recon = ReconstructFace(eig_face, rand_img, i + 1)
        if (i == 0 or i == 49 or i == 99):
            plt.subplot(121)
            plt.imshow(rand_img.reshape(193, 162), cmap='gray')
            plt.subplot(122)
            plt.imshow(recon.reshape(193, 162), cmap='gray')
            plt.show()
        mse = CalculateMSE(recon, rand_img)
        mse_img.append(mse)
    plt.plot(list(range(100)), mse_img)
    plt.xlabel("PC Num")
    plt.ylabel("Reconstruction MSE")
    plt.show()
