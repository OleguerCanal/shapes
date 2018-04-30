import cv2, math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize

class PosCalib():
    def __init__(self):
        self.point = ()


    def funct(self, params, x, y):
        k1, k2, l1, l2 = params
        fx = x*(k1 + k2*y)
        fy = y*(l1 + l2*x)
        # fx = x*k1 + k2*y
        # fy = y*l1 + l2*x
        # fx = x*(k1*x + k2*y)
        # fy = y*(l1*y + l2*x)
        return (fx, fy)

    def get_px2mm_params(self, img):
        origin_x, origin_y = self.getCoord(img)
        self.origin = origin_x, origin_y

        # mc = [[0, 0, 0, 0]]
        # image_list = [
        #     (0, 7.3),
        #     (0, 18.5),
        #     (0, 25.38),
        #     (11.12, 0),
        #     (11.12, 7.3),
        #     (11.12, 18.5),
        #     (11.12, 25.38),
        #     (20.02, 0),
        #     (20.02, 7.3),
        #     (20.02, 18.5),
        #     (20.02, 25.38),
        # ]
        #
        # for i in range(11):
        #     x1, y1 = self.getCoord(img)
        #     print x1, y1
        #     x1 -= origin_x
        #     y1 -= origin_y
        #     print x1, y1
        #     # fx = abs(float(raw_input("Enter f_x:")))
        #     # fy = abs(float(raw_input("Enter f_y:")))
        #     fx, fy = image_list[i]
        #     coord_row = [x1, y1, fx, fy]
        #     mc.append(coord_row)

        mc = [[0, 0, 0, 0], [-6.4516129032257368, 113.54838709677418, 0, 7.3], [-29.677419354838662, 325.16129032258056, 0, 18.5], [-14.193548387096712, 387.09677419354836, 0, 25.38], [158.70967741935488, 34.838709677419359, 11.12, 0], [152.25806451612908, 139.35483870967741, 11.12, 7.3], [143.22580645161298, 317.41935483870969, 11.12, 18.5], [149.67741935483878, 427.09677419354836, 11.12, 25.38], [268.38709677419365, 56.774193548387103, 20.02, 0], [263.22580645161298, 157.41935483870969, 20.02, 7.3], [256.77419354838719, 308.38709677419348, 20.02, 18.5], [256.77419354838719, 412.90322580645153, 20.02, 25.38]]

        print mc

        def eq_sys(params):
            dif = 0
            for (x, y, fx, fy) in mc:
                (fp_x, fp_y) = self.funct(params, x, y)
                dif += np.linalg.norm((fp_x-fx, fp_y-fy))
            return dif

        # As optimization problem
        x0 = (1, .0, 1, .0)
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
    print pc.origin
    x = 480
    y = 640
    print cts
    print pc.funct(cts, x, y)
