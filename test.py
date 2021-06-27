import cv2
import numpy as np

def patchRegion(i, j):
    h_start = i * 8
    h_end = h_start + 8
    w_start = j * 8
    w_end = w_start + 8
    return h_start, h_end, w_start, w_end

def transform(img, qmat):
    ''''''
    rows, cols = img.shape[:2]

    # Number of 8x8 blocks
    r8 = rows // 8; c8 = cols // 8

    # Padding image
    if rows % 8 != 0:
        img = np.lib.pad(img, ((0, 8 - rows % 8), (0, 0)), 'constant', constant_values=((0, 0), (0, 0)))
        r8 += 1

    if cols % 8 != 0:
        img = np.lib.pad(img, ((0, 0), (0, 8 - cols % 8)), 'constant', constant_values=((0, 0), (0, 0)))
        c8 += 1

    img = img.astype("float")
    dct_coefficient = np.zeros((rows//2, cols//2))

    # Calculate DCT coefficients for patches
    for i in range(r8):
        for j in range(c8):
            r_start, r_end, c_start, c_end = patchRegion(i, j)
            patch = img[r_start : r_end, c_start : c_end]
            cv2.dct(patch, patch)
            # Quantization
            patch = np.round(patch / qmat)
            dct_coefficient[r_start//2 : r_end//2, c_start//2 : c_end//2] = patch[:4, :4]
    return dct_coefficient.astype(np.int8)

def inverse_transform(dct_coefficient, qmat):
    rows, cols = dct_coefficient.shape[:2]
    rows *= 2; cols *=2
    dct_cp = dct_coefficient.copy().astype(np.float64)
    # Number of 8x8 blocks
    r8 = rows // 8; c8 = cols // 8

    result = np.zeros((rows, cols))

    # Convert back
    for i in range(r8):
        for j in range(c8):
            r_start, r_end, c_start, c_end = patchRegion(i, j)
            patch = np.zeros((8,8))
            patch[:4, :4] = dct_cp[r_start//2 : r_end//2, c_start//2 : c_end//2]
            cv2.idct(patch, patch)
            result[r_start : r_end, c_start : c_end] = patch

    # Remove padding effect
    result = result[0 : rows, 0 : cols]
    # Scale image
    pixel_max = np.max(result); pixel_min = np.min(result)
    result = (result - pixel_min)/(pixel_max - pixel_min)*255

    return result.astype(np.uint8)

img = cv2.imread("images/lena.png",0)
mat = np.array([[ 16,  11,  10,  16,  24,  40,  51,  61],
                [ 12,  12,  14,  19,  26,  58,  60,  55],
                [ 14,  13,  16,  24,  40,  57,  69,  56],
                [ 14,  17,  22,  29,  51,  87,  80,  62],
                [ 18,  22,  37,  56,  68, 109, 103,  77],
                [ 24,  35,  55,  67,  81, 104, 113,  92],
                [ 49,  64,  78,  87, 103, 121, 120, 101],
                [ 72,  92,  95,  98, 112, 100, 103,  99]])

tran = transform(img, mat)
out = inverse_transform(tran, mat)
out = cv2.GaussianBlur(out, (5,5), 0)
cv2.imshow("", out)
cv2.waitKey(0)
