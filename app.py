import numpy as np
import cv2

def lanczos_resize(img, scale_factor):
    # resize img by a scale_factor using Lanczos interpolation
    return cv2.resize(img, None, 
                      fx=scale_factor, fy=scale_factor,
                      interpolation=cv2.INTER_LANCZOS4)

def read_img(filename, monochrome=False):
    # read image
    if monochrome:
        return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
    else:
        return cv2.imread(filename)

def laplacian_filter(img):
    # Laplacian sharpening filter
    return cv2.Laplacian(img, -1, ksize=5, scale=1, 
                          delta=0, borderType=cv2.BORDER_DEFAULT)

# load and resize
img = lanczos_resize(read_img("img.jpg", monochrome=True), 2)

# scale from 0-255 to -1, 1
img = img/255.
img = 2 * img -1.

# get high frequency part and raise it to the third power
img_hf = (laplacian_filter(img))**3

# normalize
img_hf /= np.max(img_hf)

# sum up
img_sr = img + 1.5 * img_hf

# move back to -1., 1. range
img_sr = (img_sr + 1.)/2.