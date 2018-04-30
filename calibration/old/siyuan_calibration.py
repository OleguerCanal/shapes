import numpy as np
import cv2
import matplotlib.pyplot as plt
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def calibration(img,background):
    M = np.load('warp_matrix.npy')
    rows,cols,cha = img.shape
    imgw = cv2.warpPerspective(img, M, (cols, rows))
    imgwc = imgw[12:,71:571,:]
    bg_imgw = cv2.warpPerspective(background, M, (cols, rows))
    bg_imgwc = bg_imgw[12:,71:571,:]
    img_blur = cv2.GaussianBlur(bg_imgwc.astype(np.float32),(25,25),15)
    img_bs = imgwc - img_blur + np.mean(img_blur) +40
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))
    blur = 255-rgb2gray(img_blur)
    blur = blur/np.max(blur)*(0.2+(np.mean(img_blur)/np.mean(img_blur)-1)*0.8)
    blur_op = 1-blur
    cl1 = clahe.apply(img_bs[:,:,0].astype(np.uint8))
    cl2 = clahe.apply(img_bs[:,:,1].astype(np.uint8))
    cl3 = clahe.apply(img_bs[:,:,2].astype(np.uint8))
    red = img_bs[:,:,0]*blur_op + cl1*blur
    green = img_bs[:,:,1]*blur_op + cl2*blur
    blue = img_bs[:,:,2]*blur_op + cl3*blur
    im_calibrated = np.dstack((red,green,blue))
    return im_calibrated

if __name__ == "__main__":
    img = cv2.imread('sample_data/f1.jpg')
    img_back = cv2.imread('sample_data/f1_back.jpg')
    im_cal = calibration(img, img_back).astype(np.uint8)
    plt.figure()
    plt.imshow(img)
    plt.show()
    plt.figure()
    plt.imshow(im_cal)
    plt.show()
