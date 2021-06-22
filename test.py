'''
    ENGG5104 1-1 JPEG Compression
    @daibo
    @1155053920
'''
import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm as cm

def patchRegion(i, j):
    h_start = i * 8
    h_end = h_start + 8
    w_start = j * 8
    w_end = w_start + 8
    return h_start, h_end, w_start, w_end

def jpegCompress(image, quantmatrix):
    '''
        Compress(imagefile, quanmatrix simulates the lossy compression of
        baseline JPEG, by quantizing the DCT coefficients in 8x8 blocks
    '''
    # Return compressed image in result

    H = np.size(image, 0)
    W = np.size(image, 1)

    # Convert to gray-scale image
    if np.size(image, 2) > 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Number of 8x8 blocks in the height and width directions
    h8 = int(H / 8)
    w8 = int(W / 8)
    iH = H
    iW = W

    # Padding image
    if H % 8 != 0:
        image = np.lib.pad(image, ((0, 8 - H % 8), (0, 0)), 'constant', constant_values=((0, 0), (0, 0)))
        h8 += 1

    if W % 8 != 0:
        image = np.lib.pad(image, ((0, 0), (0, 8 - W % 8)), 'constant', constant_values=((0, 0), (0, 0)))
        w8 += 1

    image = image.astype("float")
    dct_coefficient = image.copy()
    result = image.copy()
    # Calculate DCT coefficients for patches
    for i in range(h8):
        for j in range(w8):
            h_start, h_end, w_start, w_end = patchRegion(i, j)
            patch = image[h_start : h_end, w_start : w_end]
            cv2.dct(patch, patch)
            # Quantization
            patch = np.round(patch / quantmatrix)
            dct_coefficient[h_start : h_end, w_start : w_end] = patch

    return dct_coefficient


    # Convert back
    for i in range(h8):
        for j in range(w8):
            h_start, h_end, w_start, w_end = patchRegion(i, j)
            patch = dct_coefficient[h_start : h_end, w_start : w_end]
            cv2.idct(patch, patch)
            result[h_start : h_end, w_start : w_end] = patch

    # Remove padding effect
    result = result[0 : iH, 0 : iW]

    return result, dct_coefficient

if __name__ == '__main__':

    im = cv2.imread('images/lena.png')

    quantmatrix = sio.loadmat('quantmatrix.mat')['quantmatrix']
    print(quantmatrix)

    dct_coefficient = jpegCompress(im, quantmatrix)

    plt.figure("JPEG Compression")

    plt.subplot(131)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    # plt.subplot(132)
    # plt.imshow(out, cmap='gray')
    # plt.title("Compressed Image")

    plt.subplot(133)
    plt.imshow(dct_coefficient, cmap='gray')
    plt.title("DCT Image")

    plt.show()

