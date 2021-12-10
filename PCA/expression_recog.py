import numpy as np
import eigenface as eigf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    smile = eigf.ReadDataset('b')
    neutral = eigf.ReadDataset('a')

    train_smile = smile[:100]
    train_neutral = neutral[:100]
    # WLOG choose the first 60 as testing set
    test_smile = smile[100:161]
    test_neutral = neutral[100:161]

    # Generate Eigen Face Space
    mean_smile, eigen_smile = eigf.CalculateEigenface(train_smile)
    mean_neutral, eigen_neutral = eigf.CalculateEigenface(train_neutral)

    # Reconstruct Image and determine if it's smiling or neutral
    correct = 0
    for i in range(60):
        cur_img = test_smile[i]
        # Neutral faces space
        recon_neutral = eigf.ReconstructFace(eigen_neutral, cur_img - mean_neutral, 100)
        neutral_mse = eigf.CalculateMSE(recon_neutral, cur_img - mean_neutral)
        # Smiling faces space
        recon_smile = eigf.ReconstructFace(eigen_smile, cur_img - mean_smile, 100)
        smile_mse = eigf.CalculateMSE(recon_smile, cur_img - mean_smile)

        if smile_mse < neutral_mse:
            correct = correct + 1
        else:
            plt.subplot(131)
            plt.imshow(cur_img.reshape(193, 162), cmap='gray')
            plt.subplot(132)
            plt.imshow(recon_neutral.reshape(193, 162), cmap='gray')
            plt.subplot(133)
            plt.imshow(recon_smile.reshape(193, 162), cmap='gray')
            # plt.show()
    print("Classification Accuracy Rate For Smiling is:", float(correct) / 60.0)
    
    correct = 0
    for i in range(60):
        cur_img = test_neutral[i]
        # Neutral faces space
        recon_neutral = eigf.ReconstructFace(eigen_neutral, cur_img - mean_neutral, 100)
        neutral_mse = eigf.CalculateMSE(recon_neutral, cur_img - mean_neutral)
        # Smiling faces space
        recon_smile = eigf.ReconstructFace(eigen_smile, cur_img - mean_smile, 100)
        smile_mse = eigf.CalculateMSE(recon_smile, cur_img - mean_smile)

        if smile_mse > neutral_mse:
            correct = correct + 1
        else:
            plt.subplot(131)
            plt.imshow(cur_img.reshape(193, 162), cmap='gray')
            plt.subplot(132)
            plt.imshow(recon_neutral.reshape(193, 162), cmap='gray')
            plt.subplot(133)
            plt.imshow(recon_smile.reshape(193, 162), cmap='gray')
            # plt.show()
    
    print("Classification Accuracy Rate For Neutral is:", float(correct) / 60.0)