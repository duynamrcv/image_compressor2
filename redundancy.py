import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_redundancy(img):
    rows, cols = img.shape
    rows_red = np.zeros((rows, cols), dtype=np.int16)
    out = np.zeros((rows, cols), dtype=np.int16)
    for i in range(rows):
        if i == 0:
            rows_red[i,:] = img[i,:]
        else:
            rows_red[i,:] = img[i,:] - img[i-1,:]
    for j in range(cols):
        if j == 0:
            out[:,j] = rows_red[:,j]
        else:
            out[:,j] = rows_red[:,j] - rows_red[:,j-1]
    return out

def get_origin(red):
    rows, cols = red.shape
    cols_red = np.zeros((rows, cols), dtype=np.uint8)
    out = np.zeros((rows, cols), dtype=np.uint8)

    for j in range(cols):
        if j == 0:
            cols_red[:,j] = red[:,j]
        else:
            cols_red[:,j] = red[:,j] + cols_red[:,j-1]

    for i in range(rows):
        if i == 0:
            out[i,:] = cols_red[i,:]
        else:
            out[i,:] = cols_red[i,:] + out[i-1,:]
    return out

# img = cv2.imread("images/lena.png", 0)
# red = get_redundancy(img)
# # print(red)

# out = get_origin(red)

# print(np.sum(out - img))

# plt.subplot(131)
# plt.imshow(img, cmap='gray')
# plt.subplot(132)
# plt.imshow(red, cmap='gray')
# plt.subplot(133)
# plt.imshow(out, cmap='gray')
# plt.show()