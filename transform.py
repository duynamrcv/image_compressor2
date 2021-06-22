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
    dct_coefficient = img.copy()

    # Calculate DCT coefficients for patches
    for i in range(r8):
        for j in range(c8):
            r_start, r_end, c_start, c_end = patchRegion(i, j)
            patch = img[r_start : r_end, c_start : c_end]
            cv2.dct(patch, patch)
            # Quantization
            patch = np.round(patch / qmat)
            dct_coefficient[r_start : r_end, c_start : c_end] = patch
    
    return dct_coefficient

def inverse_transform(dct_coefficient, qmat):
    rows, cols = dct_coefficient.shape[:2]
    dct_cp = dct_coefficient.copy()

    # Number of 8x8 blocks
    r8 = int(rows / 8)
    c8 = int(cols / 8)

    result = dct_coefficient.copy()

    # Convert back
    for i in range(r8):
        for j in range(c8):
            r_start, r_end, c_start, c_end = patchRegion(i, j)
            patch = dct_cp[r_start : r_end, c_start : c_end]
            cv2.idct(patch, patch)
            result[r_start : r_end, c_start : c_end] = patch

    # Remove padding effect
    result = result[0 : rows, 0 : cols]
    return result