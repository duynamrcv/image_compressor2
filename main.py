import cv2
import numpy as np
import matplotlib.pyplot as plt

import transform as tf

if __name__ == "__main__":
    # Read image
    img = cv2.imread("images/lena.png")

    # Convert image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Quantization matrix
    qmat = np.array([   [ 16,  11,  10,  16,  24,  40,  51,  61],
                        [ 12,  12,  14,  19,  26,  58,  60,  55],
                        [ 14,  13,  16,  24,  40,  57,  69,  56],
                        [ 14,  17,  22,  29,  51,  87,  80,  62],
                        [ 18,  22,  37,  56,  68, 109, 103,  77],
                        [ 24,  35,  55,  67,  81, 104, 113,  92],
                        [ 49,  64,  78,  87, 103, 121, 120, 101],
                        [ 72,  92,  95,  98, 112, 100, 103,  99]])

    ################
    ### Encoding ###
    ################

    # Tranformation and Quantization
    dct_coefficient = tf.transform(gray, qmat)
    
    ################
    ### Decoding ###
    ################

    # De-Tranformation and De-Quantization
    out = tf.inverse_transform(dct_coefficient, qmat)

    # Visulization
    plt.subplot(121)
    plt.imshow(gray, cmap='gray'); plt.axis('off')
    plt.title("Original Image")
    plt.subplot(122)
    plt.imshow(out, cmap='gray'); plt.axis('off')
    plt.title("Reconstructed Image"
    plt.show()