import cv2, math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

class PosCalib():
    def __init__(self):
        self.point = ()

    def get_px2mm_params(self, img):
        # Idea:
        # Repeat 4 times:
        # 1. Record 2 points
        # 2. Entrar distancia real
        # 3. Solve system

        def get_eq_coefs(x1, y1, x2, y2):
            alfa = float(x1 - x2)
            beta = float(x1*y1 - x2*y2)
            gamma = float(y1 - y2)
            a = alfa ** 2
            b = beta ** 2
            c = 2*alfa*beta
            d = gamma ** 2
            e = 2*beta*gamma
            return a, b, c, d, e

        mat = []
        vect = []
        for i in range(4):
            x1, y1 = self.getCoord(img)
            x2, y2 = self.getCoord(img)
            real_dist = abs(float(raw_input("Enter real distance:")))
            a, b, c, d, e = get_eq_coefs(x1, y1, x2, y2)
            param_row = [a, c, b, d, e, b]
            mat.append(param_row)
            vect.append(real_dist)

        def eq_system(params):
            k1, k2, l1, l2 = params
            f1 = vect[0]**2 - (mat[0][0]*k1**2 + mat[0][1]*k1*k2 + mat[0][2]*k2**2 + mat[0][3]*l1**2 + mat[0][4]*l1*l2 + mat[0][5]*l2**2)
            f2 = vect[1]**2 - (mat[1][0]*k1**2 + mat[1][1]*k1*k2 + mat[1][2]*k2**2 + mat[1][3]*l1**2 + mat[1][4]*l1*l2 + mat[1][5]*l2**2)
            f3 = vect[2]**2 - (mat[2][0]*k1**2 + mat[2][1]*k1*k2 + mat[2][2]*k2**2 + mat[2][3]*l1**2 + mat[2][4]*l1*l2 + mat[2][5]*l2**2)
            f4 = vect[3]**2 - (mat[3][0]*k1**2 + mat[3][1]*k1*k2 + mat[3][2]*k2**2 + mat[3][3]*l1**2 + mat[3][4]*l1*l2 + mat[3][5]*l2**2)
            return (f1, f2, f3, f4)

        params = (0.1, 0.05, .1, 0.05)
        k1, k2, l1, l2 = fsolve(eq_system, params)
        return [k1, k2, l1, l2]

    def getCoord(self, img):
        fig = plt.figure()
        plt.imshow(img)
        cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()
        return (img.shape[1]-self.point[1], self.point[0])  # Fix axis values

    def __onclick__(self, click):
        self.point = (click.xdata, click.ydata)
        plt.close('all')
        return self.point


class DepthCalib():
    def __init__(self):
        pass

    def multiple_force_ball_calibration(self, img_list, force_list):
        pass


if __name__ == "__main__":
    pc = PosCalib()
    path = 'sample_data/pos_calibration/p_0/GS1_0.png'
    print pc.get_px2mm_params(cv2.imread(path))
