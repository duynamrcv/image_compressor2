import cv2
import numpy as np
import matplotlib.pyplot as plt

def mse(ori, rec):
    rows, cols = ori.shape
    return 1/(rows*cols)*(np.sum((ori - rec)**2))

def psnr(ori, rec):
    mse_value = mse(ori, rec)
    return 10*(np.log10(255**2/mse_value))

def compress_ratio(ori, data):
    rows, cols = ori.shape[:2]
    v_ori = rows*cols*8
    v_data = len(data)*8.2
    return v_ori/v_data