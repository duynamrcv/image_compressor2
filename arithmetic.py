import cv2
import numpy as np
import time

def probability(mat):
    rows, cols = mat.shape
    total = rows*cols
    freq = {} 
    for i in range(rows):
        for j in range(cols):
            item = mat[i,j]
            if (item in freq):
                freq[item] += 1
            else:
                freq[item] = 1
    for key, value in freq.items():
        freq[key] = value / total
    return freq

def create_range(freq):
    range_left = {}; range_right = {}
    start = 0
    for k in freq.keys():
        range_left[k] = start
        range_right[k] = start + freq[k]
        start = start + freq[k]
    return range_left, range_right

def compress(mat, freq, range_left, range_right):
    block_size=4
    rows, cols = mat.shape
    total = rows*cols
    mat_1d = mat.reshape(total)
    res = np.zeros(total//block_size, dtype=np.float64)  # final codes
    for i in range(0, total, block_size):
        l = 0.0; r = 1.0

        for j in range(i, i + block_size):
            oldLeft = l; oldRight = r
            # if mat_1d[j] != 0:
            l = oldLeft + (oldRight - oldLeft) * range_left[mat_1d[j]]
            r = oldLeft + (oldRight - oldLeft) * range_right[mat_1d[j]]
        # result of the block is the average of (upper - lower)
        it = int(i/block_size)
        res[it] = (l + r)/2
    return res

def decompress(res, freq, range_left, range_right, shape):
    block_size=4
    rows, cols = shape
    total = rows * cols
    out = np.zeros(total)

    for i in range(0, total, block_size):
        l = 0.0; r = 1.0
        code = res[i//block_size]
        for j in range(i, i + block_size):
            for k in freq.keys():
                # if code >= l + (r-l)*freq[k]:
                oldLeft = l; oldRight = r
                lCheck = oldLeft + (oldRight - oldLeft)*range_left[k]
                rCheck = oldLeft + (oldRight - oldLeft)*range_right[k]
                if code > lCheck and code < rCheck:
                    out[j] = k
                    l = lCheck; r = rCheck
                    break

    out = np.array(out).reshape((rows, cols))
    return out.astype(np.int16)

# img = np.array([[ 16,  11,  10,  16,  24,  40,  51,  61],
#                 [ 12,  12,  14,  19,  26,  58,  60,  55],
#                 [ 14,  13,  16,  24,  40,  57,  69,  56],
#                 [ 14,  17,  22,  29,  51,  87,  80,  62],
#                 [ 18,  22,  37,  56,  68, 109, 103,  77],
#                 [ 24,  35,  55,  67,  81, 104, 113,  92],
#                 [ 49,  64,  78,  87, 103, 121, 120, 101],
#                 [ 72,  92,  95,  98, 112, 100, 103,  99]])

# img = cv2.imread("images/lena.png", 0)

# freq = probability(img)
# range_left, range_right = create_range(freq)
# block_size=4

# # for k in freq.keys():
# #     print("Value: {}\t Prob: {}\t Left: {}\t Right: {}".format(k, freq[k], range_left[k], range_right[k]))
# begin = time.time()
# print("Compressing")
# res = compress(img, freq, range_left, range_right, block_size=block_size)
# finish = time.time()
# print("Done")
# print("{:.2f}s".format(finish - begin))

# begin = time.time()
# print("Decompressing")
# out = decompress(res, freq, range_left, range_right, img.shape, block_size=block_size)
# finish = time.time()
# print("Done")
# print("{:.2f}s".format(finish - begin))

# # print(img)
# # print(out)
# # print(out-img)

# cv2.imshow("In", img)
# cv2.imshow("Out", out)
# cv2.waitKey(0)
