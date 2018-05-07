import numpy as np
from image_processing import ImageProcessing
from matplotlib import pyplot as plt
import cv2, math, scipy.io
from PIL import Image
from scipy.misc import toimage

n_x = 480
n_y = 640

def plot(f):
    im = plt.imshow(f, cmap='hot')
    plt.colorbar(im, orientation='horizontal')
    plt.show()

class Labeller():
    def __init__(self):
        self.ip = ImageProcessing()
        self.params_gs1 = [0.0503, -0.0026, 0,    -0.0699, -0.0038, 0,   6.5039, 6.4795, 16.713]
        self.real_r = 6.35/2

    def __distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        # return np.linalg.norm((x1-x2, y1-y2))
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    def __get_mm_pos(self, point):
        x, y = point
        k1, k2, k3,  l1, l2, l3,   dx, dy, dz = self.params_gs1
        p1 = (x, y - 640.0/2)
        return (p1[0]*k1 + p1[1]*k2 + k3*p1[0]**2, p1[1]*l1 + p1[0]*l2 + l3*p1[1]**2)

    def __mm_2_px(self, dist):
        return 10*dist

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
        for x_pixel in range(n_x):
            for y_pixel in range(n_y):
                if self.__distance((x_pixel, y_pixel), center) < min(radius, pixel_r):
                    # x, y = self.__get_mm_pos((x_pixel, y_pixel))
                    # x_dif = (x - c_x)
                    # y_dif = (y - c_y)
                    try:
                        # dz_dx = -x_dif/(math.sqrt(self.real_r**2 - x_dif**2 - y_dif**2))
                        # dz_dy = -y_dif/(math.sqrt(self.real_r**2 - x_dif**2 - y_dif**2))
                        x_dif = (x_pixel - center[0])
                        y_dif = (y_pixel - center[1])
                        # dz_dx = -x_dif/(math.sqrt(radius**2 - x_dif**2 - y_dif**2))
                        # dz_dy = -y_dif/(math.sqrt(radius**2 - x_dif**2 - y_dif**2))

                        dz_dx = -x_dif/(math.sqrt(pixel_r**2 - x_dif**2 - y_dif**2))
                        dz_dy = -y_dif/(math.sqrt(pixel_r**2 - x_dif**2 - y_dif**2))

                        dz_dx_mat[x_pixel][y_pixel] = dz_dx
                        dz_dy_mat[x_pixel][y_pixel] = dz_dy
                    except:
                        print "error"
                        pass

        return dz_dx_mat, dz_dy_mat

    def shape_reconstruction(self, dz_dx, dz_dy):
        n_x = 480
        n_y = 640

        # We integrate dz_dx WRT x
        int_dx = dz_dx
        for i in range(n_x-1):
            for j in range(n_y):
                int_dx[i+1][j] += int_dx[i][j]
        plot(int_dx)

        # We derive int_dx WRT y
        dy_int_dx = int_dx
        for i in range(n_x):
            for j in range(n_y-1):
                dy_int_dx[i][j] = int_dx[i][j+1] - int_dx[i][j]
        plot(dy_int_dx)

        # We substract dy_int_dx - dz_dy to get d_h
        dh_dy = np.zeros((n_x, n_y))
        for i in range(n_x):
            for j in range(n_y):
                dh_dy[i][j] = dy_int_dx[i][j] - dz_dy[i][j]
        plot(dh_dy)

        # We integrate dh_dy WRT y
        h = dh_dy
        for i in range(n_x):
            for j in range(n_y-1):
                jj = j + 1
                h[i][jj] += h[i][jj-1]
        plot(h)

        f = np.zeros((n_x, n_y))
        for i in range(n_x):
            for j in range(n_y):
                f[i][j] = int_dx[i][j] - h[i][j]
        plot(f)
        return f

    def labell_image(self, img, img_back):
        # self.ip.show_2images(img, img_back)

        no_back = self.ip.crop_contact(
            img_back=img,
            img_grasp=img_back,
            gel_id=1,
            compress_factor=1,
            is_zeros=True)

        # cv2.imshow("Keypoints", no_back)
        # cv2.waitKey(0)

        ret, thresh = cv2.threshold(no_back,127,255,0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        cnt = contours[0]
        # M = cv2.moments(cnt)

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        cv2.circle(no_back, center, radius, (200), 2)
        cv2.imshow("no_back", no_back)
        cv2.waitKey(0)

        print center, radius
        shape = np.shape(no_back)

        dz_dx_mat, dz_dy_mat = self.get_gradient_matrices(center=center, radius=radius, size=shape)
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
