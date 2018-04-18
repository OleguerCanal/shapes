import cv2, math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize

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
        mc = []
        vect = []

        # mc = [[30.467741935483787, 93.693548387096769, 285.95161290322574, 511.75806451612902], [17.564516129032086, 213.6935483870968, 305.30645161290317, 162.08064516129033], [195.6290322580644, 131.11290322580646, 291.11290322580635, 408.5322580645161], [11.112903225806349, 488.5322580645161, 176.27419354838702, 414.98387096774195]]
        # vect = [32.33, 21.31, 20.53, 13.08]

        mc = [[30.467741935483787, 93.69354838709677, 285.95161290322574, 511.758064516129], [17.564516129032086, 213.6935483870968, 305.3064516129032, 162.08064516129033], [195.6290322580644, 131.11290322580646, 291.11290322580635, 408.5322580645161], [11.112903225806349, 488.5322580645161, 176.27419354838702, 414.98387096774195], [31.758064516128911, 100.14516129032259, 189.17741935483861, 242.08064516129033], [7.2419354838708614, 409.82258064516122, 186.5967741935483, 242.08064516129033], [193.04838709677409, 133.69354838709677, 296.27419354838702, 261.43548387096769], [186.5967741935483, 244.66129032258064, 291.11290322580635, 408.5322580645161]]
        vect = [32.33, 21.31, 20.53, 13.08, 13.3, 15.84, 11.51, 14.3]

        # for i in range(4):
        #     x1, y1 = self.getCoord(img)
        #     print x1, y1
        #     x2, y2 = self.getCoord(img)
        #     print x2, y2
        #     real_dist = abs(float(raw_input("Enter real distance:")))
        #     # a, b, c, d, e = get_eq_coefs(x1, y1, x2, y2)
        #     # param_row = [a, c, b, d, e, b]
        #     coord_row = [x1, y1, x2, y2]
        #     # mat.append(param_row)
        #     mc.append(coord_row)
        #     vect.append(real_dist)

        self.mc = mc
        self.dist = vect



        def __sq_distance_aprox(params, x1, y1, x2, y2):
            k1, k2, l1, l2 = params
            # dist_x = x1*(k1*x1 + k2*y1) - x2*(k1*x2 + k2*y2)
            # dist_y = y1*(l1*y1 + l2*x1) - y2*(l1*y2 + l2*x2)
            dist_x = x1*(k1 + k2*y1) - x2*(k1 + k2*y2)
            dist_y = y1*(l1 + l2*x1) - y2*(l1 + l2*x2)
            return dist_x**2 + dist_y**2

        def eq_sys(params):
            dif = 0
            for (x1, y1, x2, y2), distance in zip(self.mc, self.dist):
                dif += abs(distance**2 - __sq_distance_aprox(params, x1, y1, x2, y2))
            return dif

        def eq_system(params):
            k1, k2, l1, l2 = params
            f1 = vect[0]**2 - (mat[0][0]*k1**2 + mat[0][1]*k1*k2 + mat[0][2]*k2**2 + mat[0][3]*l1**2 + mat[0][4]*l1*l2 + mat[0][5]*l2**2)
            f2 = vect[1]**2 - (mat[1][0]*k1**2 + mat[1][1]*k1*k2 + mat[1][2]*k2**2 + mat[1][3]*l1**2 + mat[1][4]*l1*l2 + mat[1][5]*l2**2)
            f3 = vect[2]**2 - (mat[2][0]*k1**2 + mat[2][1]*k1*k2 + mat[2][2]*k2**2 + mat[2][3]*l1**2 + mat[2][4]*l1*l2 + mat[2][5]*l2**2)
            f4 = vect[3]**2 - (mat[3][0]*k1**2 + mat[3][1]*k1*k2 + mat[3][2]*k2**2 + mat[3][3]*l1**2 + mat[3][4]*l1*l2 + mat[3][5]*l2**2)
            return (f1, f2, f3, f4)

        def eq_system2(params):
            k1, k2, l1, l2 = params
            f1 = vect[0]**2 - ((k1 + k2*mc[0][1])*mc[0][0] - (k1 + k2*mc[0][3])*mc[0][2])**2 - ((l1 + l2*mc[0][0])*mc[0][1] - (l1 + l2*mc[0][2])*mc[0][3])**2
            f2 = vect[1]**2 - ((k1 + k2*mc[1][1])*mc[1][0] - (k1 + k2*mc[1][3])*mc[1][2])**2 - ((l1 + l2*mc[1][0])*mc[1][1] - (l1 + l2*mc[1][2])*mc[1][3])**2
            f3 = vect[2]**2 - ((k1 + k2*mc[2][1])*mc[2][0] - (k1 + k2*mc[2][3])*mc[2][2])**2 - ((l1 + l2*mc[2][0])*mc[2][1] - (l1 + l2*mc[2][2])*mc[2][3])**2
            f4 = vect[3]**2 - ((k1 + k2*mc[3][1])*mc[3][0] - (k1 + k2*mc[3][3])*mc[3][2])**2 - ((l1 + l2*mc[3][0])*mc[3][1] - (l1 + l2*mc[3][2])*mc[3][3])**2
            print (f1, f2, f3, f4)
            print (k1, k2, l1, l2)
            return (f1, f2, f3, f4)

        def eq_system3(params):
            k1, k2, l1, l2 = params
            f1 = vect[0]**2 - ((k1*mc[0][0] + k2*mc[0][1]) - (k1*mc[0][2] + k2*mc[0][3]))**2 - ((l1*mc[0][1] + l2*mc[0][0]) - (l1*mc[0][3] + l2*mc[0][2]))**2
            f2 = vect[1]**2 - ((k1*mc[1][0] + k2*mc[1][1]) - (k1*mc[1][2] + k2*mc[1][3]))**2 - ((l1*mc[1][1] + l2*mc[1][0]) - (l1*mc[1][3] + l2*mc[1][2]))**2
            f3 = vect[2]**2 - ((k1*mc[2][0] + k2*mc[2][1]) - (k1*mc[2][2] + k2*mc[2][3]))**2 - ((l1*mc[2][1] + l2*mc[2][0]) - (l1*mc[2][3] + l2*mc[2][2]))**2
            f4 = vect[3]**2 - ((k1*mc[3][0] + k2*mc[3][1]) - (k1*mc[3][2] + k2*mc[3][3]))**2 - ((l1*mc[3][1] + l2*mc[3][0]) - (l1*mc[3][3] + l2*mc[3][2]))**2
            # print (f1, f2, f3, f4)
            return (f1, f2, f3, f4)

        print mc
        print vect
        # params = (0.1, 1, 0.07, 1)
        # params = (10, 10.0, 10.0, 10.0)
        # k1, k2, l1, l2 = fsolve(eq_system2, params)

        # As optimization problem
        x0 = (0.2, .001, 0.2, .001)
        res = minimize(eq_sys, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
        (k1, k2, l1, l2) = res.x

        return [k1, k2, l1, l2]

    def getCoord(self, img):
        fig = plt.figure()
        plt.imshow(img)
        cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()
        return (self.point[1], self.point[0])  # Fix axis values

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
    path = 'pos_calibration/GS1_sq_calib.png'
    cts = pc.get_px2mm_params(cv2.imread(path))
    # cts = [0.087504891478981175, -0.00012331405816522298, 0.010029617758599094, 0.00019075236567386814]
    x = 480
    y = 640
    print cts
    mm = ((cts[0] + cts[1]*y)*x, (cts[2] + cts[3]*x)*y)
    print mm
    print math.sqrt(mm[0]**2 + mm[1]**2)

    # Standard params:
    # [0.087504891478981175, -0.00012331405816522298, 0.010029617758599094, 0.00019075236567386814]
