import cv2
import numpy as np
import matplotlib.pyplot as plt

import arithmetic as ar
import transform as tf

img = cv2.imread("images/lena.png", 0)

qmat = np.array([   [ 16,  11,  10,  16,  24,  40,  51,  61],
                    [ 12,  12,  14,  19,  26,  58,  60,  55],
                    [ 14,  13,  16,  24,  40,  57,  69,  56],
                    [ 14,  17,  22,  29,  51,  87,  80,  62],
                    [ 18,  22,  37,  56,  68, 109, 103,  77],
                    [ 24,  35,  55,  67,  81, 104, 113,  92],
                    [ 49,  64,  78,  87, 103, 121, 120, 101],
                    [ 72,  92,  95,  98, 112, 100, 103,  99]])
tran = tf.transform(img, qmat)
freq = ar.probability(tran)
range_left, range_right = ar.create_range(freq)
# for k in freq.keys():
#     print("Value: {}\t Prob: {}\t Left: {}\t Right: {}".format(k, freq[k], range_left[k], range_right[k]))
ari_code = ar.compress(tran, freq, range_left, range_right)

ari_decode = ar.decompress(ari_code, freq, range_left, range_right, img.shape)
out = tf.inverse_transform(ari_decode, qmat)

# Visualization
plt.subplot(121)
plt.imshow(img, cmap='gray'); plt.axis('off')
plt.title("Original Image")
plt.subplot(122)
plt.imshow(out, cmap='gray'); plt.axis('off')
plt.title("Reconstructed Image")
plt.show()