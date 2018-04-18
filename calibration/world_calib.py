import cv2, math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize
from world_positioning import pxb_2_wb
import os, pickle

class PosCalib():
    def __init__(self):
        self.point = ()

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

    def get_contact_info(self, directory, num):
        def get_cart(path):
            cart = np.load(path)
            a = cart[3]
            cart[3] = cart[6]
            b = cart[4]
            cart[4] = a
            c = cart[5]
            cart[5] = b
            cart[6] = c
            return cart

        def load_obj(path):
            with open(path, 'rb') as f:
                return pickle.load(f)

        directory += '/p_' + str(num)
        file_list = os.listdir(directory)

        cart = get_cart(directory + '/cart.npy')
        gs1_list = []
        gs2_list = []
        wsg_list = []

        for elem in file_list:
            path = directory + '/' + elem
            if 'GS1' in elem:
                gs1_list.append(cv2.imread(path))
            elif 'GS2' in elem:
                gs2_list.append(cv2.imread(path))
            elif 'wsg' in elem:
                wsg_list.append(load_obj(path))

        # force_list = []
        # for elem in wsg_list:
        #     force_list.append(elem['force'])

        return cart, gs1_list, gs2_list, wsg_list

    def __get_point_list(self):
        image_list = [
            (.0, .0),
            (.0, 7.3),
            # (0, 18.5),
            # (0, 25.38),
            (11.12, 0),
            (11.12, 7.3),
            (11.12, 18.5),
            (11.12, 25.38),
            (20.02, 0),
            (20.02, 7.3),
            (20.02, 18.5),
            (20.02, 25.38)
        ]
        point_list = []
        for elem in image_list:
            x = 487.56 - 7.3 + elem[0]
            y = 605.05
            z = 828.7 + 11.12 - elem[1] - (250+72.5+139.8)  # We substract the mesuring tool
            point_list.append((x, y, z))
        return point_list

    def __gather_points_pairs(self):
        path = 'pos_calibration/pos_calibration_squares'
        gs_id = 1

        mc = []
        point_list = self.__get_point_list()

        for i in range(3):
            cart, gs1_list, gs2_list, wsg_list = self.get_contact_info(path, i)

            gripper_state = {}
            gripper_state['pos'] = cart[0:3]
            gripper_state['quaternion'] = cart[-4:]
            gripper_state['Dh'] = wsg_list[0]['width']
            gripper_state['Dv'] = 139.8 + 72.5 + 160 # Base + wsg + finger

            for j in range(10):
                print "Touch point: " + str(j+1)
                if gs_id == 1:
                    point = self.getCoord(gs1_list[0])
                elif gs_id == 2:
                    point = self.getCoord(gs2_list[0])
                mc.append((point, gs_id, gripper_state, point_list[j][0], point_list[j][1], point_list[j][2]))

        return mc

    def get_px2mm_params(self):

        mc = self.__gather_points_pairs()
        # print mc

        def eq_sys(params):
            dif = 0
            n = len(mc)
            for (point, gs_id, gripper_state, fx, fy, fz) in mc:
                (fp_x, fp_y, fp_z) = pxb_2_wb(point, gs_id, gripper_state, params)
                dif += np.linalg.norm((fp_x-fx, fp_y-fy, fp_z-fz))
            return dif/n # This number is the average distance between estimations and real points

        # As optimization problem
        x0 = (0.1, .0, 0.07, .001, .0, .0)
        bounds = [
            (0, 1),
            (-.5, .5),
            (0, 1),
            (-.5, .5),
            (-500, 500),
            (-30, 30)
        ]
        res = minimize(eq_sys, x0, bounds=bounds, options={'xtol': 1e-8, 'disp': False})
        # res = minimize(eq_sys, x0, options={'xtol': 1e-8, 'disp': True})
        print res
        (k1, k2, l1, l2, dh, dv) = res.x

        return [k1, k2, l1, l2, dh, dv]

    def test(self, point, path, num, params):
        cart, gs1_list, gs2_list, wsg_list = self.get_contact_info(path, num)

        gripper_state = {}
        gripper_state['pos'] = cart[0:3]
        gripper_state['quaternion'] = cart[-4:]
        gripper_state['Dh'] = wsg_list[0]['width']
        gripper_state['Dv'] = 139.8 + 72.5 + 160  # Base + wsg + finger

        point_wb = pxb_2_wb(point, 1, gripper_state, params)
        return point_wb


if __name__ == "__main__":
    pc = PosCalib()
    path = 'pos_calibration/pos_calibration_squares'

    cts = pc.get_px2mm_params()

    # cts = [0.083553239586688935, 1.2890186744400053e-06, 0.05986013359232522, -0.00017069733889997396, 0.0053275523201169071, -0.001310254063738562]

    print cts
    point = (0, 320)
    point_wb = pc.test(point, path, 2, cts)
    print point_wb
