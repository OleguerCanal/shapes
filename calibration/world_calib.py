import cv2, math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize
from world_positioning import pxb_2_wb
import os, pickle

class PosCalib():
    def __init__(self):
        pass

    def get_px2mm_params(self):
        squares = np.load('pos_calibration/squares.npy').tolist()
        border = np.load('pos_calibration/border.npy').tolist()
        line = np.load('pos_calibration/line.npy').tolist()
        self.mc = squares + border + line
        # self.mc = border


        def eq_sys(params):
            dif = 0
            n = len(self.mc)
            for (gs_point, gs_id, gripper_state, real_point) in self.mc:
                fx, fy, fz = real_point
                (fp_x, fp_y, fp_z) = pxb_2_wb(gs_point, gs_id, gripper_state, params)
                dif += np.linalg.norm((fp_x-fx, fp_y-fy, fp_z-fz))
            return dif/n  # This number is the average distance between estimations and real points

        # Solve optimization problem
        # x0 = (.0, 0.1, .0,   .0, 0.07, .0,   1.4, -.24, -1.2)
        x0 = (0, 0,   0, 0,   0, 0, 0)
        x0 = (2, 3,   5, 0,   10, 20, -30)
        x0 = (0.0503, -0.0026, 0,    -0.0699, -0.0038, 0,   6.5039, 6.4795, 16.713)
        # x0 = (2.0, 0.2, 13.0)

        # res = minimize(eq_sys, x0, bounds=bounds, options={'xtol': 1e-8, 'disp': False})
        res = minimize(eq_sys, x0, method='Nelder-Mead', options={'xtol': 1e-8, 'disp': True})
        print "Success: " + str(res.success)
        print "Average distance: " + str(res.fun)

        return res.x

    def test_all(self, params):
        for (point, gs_id, gripper_state, real_point) in self.mc:
            fx, fy, fz = real_point
            (fp_x, fp_y, fp_z) = pxb_2_wb(point, gs_id, gripper_state, params)
            print "Real: " + str((fx, fy, fz))
            print "Guessed: " + str((fp_x, fp_y, fp_z))
            print "Diference: " + str((fp_x-fx, fp_y-fy, fp_z-fz))
            print "Distance: " + str(np.linalg.norm((fp_x-fx, fp_y-fy, fp_z-fz)))
            print "*****"


if __name__ == "__main__":
    pc = PosCalib()
    path = 'pos_calibration/pos_calibration_squares'

    cts = pc.get_px2mm_params()

    # cts = [-5.31736713e-04, 1.32689988e-01, 1.92324850e-01, 4.93343916e-04, -2.60680326e-01,  2.58946303e-02,   1.18242193e+02, 2.69041742e-01, -1.72261820e+00]
# 0.08
# -0.00
# -0.07
# -0.00
# 6.44
# 6.28
# 14.41



    for elem in cts:
        print("%.4f" % elem)

    # pc.test_all(params=cts)
    # pc.test_new_squares(params=cts)
