import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

import transform as tf
import arithmetic as ar
import quality as qa

if __name__ == "__main__":
    # Read image
    img = cv2.imread("images/cameraman.jpg")

    # Convert image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5,5), 0)

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
    en_start = time.time()

    # Tranformation and Quantization
    dct_coefficient = tf.transform(gray, qmat)

    # Entropy coding: Arithmetic
    freq = ar.probability(dct_coefficient)
    range_left, range_right = ar.create_range(freq)
    res = ar.compress(dct_coefficient, freq, range_left, range_right)

    en_finish = time.time()
    print("Encoding time: {:.2f}s".format(en_finish - en_start))
    
    ################
    ### Decoding ###
    ################

    de_start = time.time()

    # Entropy decoding: Arithmetic
    quan = ar.decompress(res, freq, range_left, range_right, dct_coefficient.shape)

    # De-Tranformation and De-Quantization
    out = tf.inverse_transform(quan, qmat)

    # # Get origin
    # out = re.get_origin(de_trans)
    de_finish = time.time()
    print("Decoding time: {:.2f}s".format(de_finish - de_start))

    ##################
    ### Compensate ###
    ##################
    out = cv2.blur(out, (3,3))
    out = cv2.GaussianBlur(out, (5,5), 0)
    blur = cv2.blur(out, (5,5))
    unsharp = out - blur
    out = out + unsharp

    ###############
    ### Quality ###
    ###############
    print("MSE: {}".format(qa.mse(gray, out)))
    print("PSNR: {}".format(qa.psnr(gray, out)))

    #################
    ### Visualize ###
    #################
    plt.subplot(121)
    plt.imshow(gray, cmap='gray'); plt.axis('off')
    plt.title("Original Image")
    plt.subplot(122)
    plt.imshow(out, cmap='gray'); plt.axis('off')
    plt.title("Reconstructed Image")
    plt.show()

    # er = gray - out
    # plt.hist(er.ravel(), 256, [0,256]); plt.axis('off')
    # plt.show()