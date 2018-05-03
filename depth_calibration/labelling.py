import numpy as np
from image_processing import ImageProcessing
import cv2, math

class Labeller():
    def __init__(self):
        self.ip = ImageProcessing()
        self.params_gs1 = [0.0503, -0.0026, 0,    -0.0699, -0.0038, 0,   6.5039, 6.4795, 16.713]
        self.real_r = 6.35

    def __distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return np.linalg.norm(x1-x2, y1-y2)

    def __get_mm_pos(self, point):
        x, y = point
        k1, k2, k3,  l1, l2, l3,   dx, dy, dz = self.params_gs1
        p1 = (x, y - 640.0/2)
        return (p1[0]*k1 + p1[1]*k2 + k3*p1[0]**2, p1[1]*l1 + p1[0]*l2 + l3*p1[1]**2)

    def get_gradient_matrices(center, radius, size=(480, 640)):
        # Everyhting given in pixel space
        n_x, n_y = size
        pixel_r = radius

        c_x, c_y = self.__get_mm_pos(center)
        # We calculate the real radius
        perim_point_mm = self.__get_mm_pos(center + (pixel_r, 0))
        r = self.__distance((c_x, c_y), perim_point_mm)

        if r > self.real_r:
            print "ERROR: Estimated radius larger than real radius!"

        dz_dx_mat = np.zeros(size)
        dz_dy_mat = np.zeros(size)
        for x_pixel in range(n_x):
            for y_pixel in range(n_y):
                if self.__distance((x_pixel, y_pixel), center) < r_pixel:
                    x, y = self.__get_mm_pos((x_pixel, y_pixel))
                    x_dif = (x - c_x)
                    y_dif = (y - c_y)
                    dz_dx = -x_dif/(math.sqrt(self.real_r**2 - x_dif**2 - y_dif**2)
                    dz_dy = -y_dif/(math.sqrt(self.real_r**2 - x_dif**2 - y_dif**2)

                    dz_dx_mat[x_pixel][y_pixel] = dz_dx
                    dz_dy_mat[x_pixel][y_pixel] = dz_dy

        return dz_dx_mat, dz_dy_mat

    def labell_image(self, img, img_back):
        # self.ip.show_2images(img, img_back)

        no_back = self.ip.crop_contact(
            img_back=img,
            img_grasp=img_back,
            gel_id=1,
            compress_factor=1,
            is_zeros=True)

        no_back = cv2.imread("foto.png")
        no_back = no_back.astype(np.uint8)

        cv2.imshow("Keypoints", no_back)
        cv2.waitKey(0)

        ret, thresh = cv2.threshold(no_back,127,255,0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        cnt = contours[0]
        # M = cv2.moments(cnt)

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        cv2.circle(no_back, center, radius, (200), 2)
        cv2.imshow("Keypoints", no_back)
        cv2.waitKey(0)

        print center, radius
        return self.get_gradient_matrices


if __name__ == "__main__":
    img = cv2.imread('air.png')
    img_back = cv2.imread('balls.png')

    labeller = Labeller()
    labeller.labell_image(img=img, img_back=img_back)
