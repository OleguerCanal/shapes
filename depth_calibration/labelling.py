import numpy as np
from matplotlib import pyplot as plt
import cv2, math, scipy.io
from PIL import Image
from scipy.misc import toimage

def plot(f):
    # im = plt.imshow(f, cmap='hot')
    # plt.colorbar(im, orientation='horizontal')
    # plt.show()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow("im2", f)
    cv2.waitKey(0)



class Labeller():
    def __init__(self):
        # self.ip = ImageProcessing()
        self.params_gs1 = [0.0503, -0.0026, 0,    -0.0699, -0.0038, 0,   6.5039, 6.4795, 16.713]
        self.real_r = 6.35/2
        self.mm2px_param = 13.1

    def __distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return np.linalg.norm((x1-x2, y1-y2))

    def __get_mm_pos(self, point):
        x, y = point
        k1, k2, k3,  l1, l2, l3,   dx, dy, dz = self.params_gs1
        p1 = (x, y - 640.0/2)
        return (p1[0]*k1 + p1[1]*k2 + k3*p1[0]**2, p1[1]*l1 + p1[0]*l2 + l3*p1[1]**2)

    def __mm_2_px(self, dist):
        return self.mm2px_param*float(dist)

    def get_gradient_matrices(self, center, radius, size):
        # Everyhting given in pixel space
        n_x, n_y = size
        pixel_r = self.__mm_2_px(self.real_r)
        (x, y) = center
        center = (y, x)

        # c_x, c_y = self.__get_mm_pos(center)
        # # We calculate the real radius
        # perim_point_px = (center[0] + pixel_r, center[1])
        # print perim_point_px
        # perim_point_mm = self.__get_mm_pos(perim_point_px)
        # r = self.__distance((c_x, c_y), perim_point_mm)
        #
        if pixel_r < radius:
            print "ERROR: Estimated radius larger than real radius!"
            print "Image Radius: " + str(radius)
            print "Real radius in px: " + str(pixel_r)

        #
        dz_dx_mat = np.zeros(size)
        dz_dy_mat = np.zeros(size)

        xvalues = np.array(range(size[0]))
        yvalues = np.array(range(size[1]))

        x_pixel, y_pixel = np.meshgrid(xvalues, yvalues)

        x_dif = (x_pixel - center[0]).astype(np.float32)
        y_dif = (y_pixel - center[1]).astype(np.float32)

        dz_dx_mat = -x_dif/(np.sqrt(np.abs(pixel_r**2 - x_dif**2 - y_dif**2)))
        dz_dy_mat = -y_dif/(np.sqrt(np.abs(pixel_r**2 - x_dif**2 - y_dif**2)))

        mask = ((x_dif**2 + y_dif**2) < min(pixel_r, radius)**2).astype(np.float32)
        dz_dx_mat = (dz_dx_mat * mask).astype(np.float32)
        dz_dy_mat = (dz_dy_mat * mask).astype(np.float32)

        # plot(dz_dx_mat)
        # plot(dz_dy_mat)
        return dz_dx_mat, dz_dy_mat


if __name__ == "__main__":
    img = cv2.imread('data/GS1_1.png')
    img_back = cv2.imread('data/GS1_20.png')

    labeller = Labeller()
    dz_dx_mat, dz_dy_mat = labeller.labell_image(img=img, img_back=img_back)
    # for i in range(n_x):
    #     for j in range(n_y):
    #         if abs(dz_dx_mat[i][j]) > 5:
    #             dz_dx_mat[i][j] = 5*np.sign(dz_dx_mat[i][j])
    plot(dz_dx_mat)
    plot(dz_dy_mat)

    # f = labeller.shape_reconstruction(dz_dx_mat, dz_dy_mat)
    # scipy.io.savemat('f.mat', mdict={'arr': f})



    scipy.io.savemat('dz_dx_mat.mat', mdict={'arr': dz_dx_mat})
    scipy.io.savemat('dz_dy_mat.mat', mdict={'arr': dz_dy_mat})
