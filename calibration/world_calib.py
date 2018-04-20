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
            x = 485.39 + elem[1]
            y = 607.03
            z = 378.3 - elem[0] # - (250+72.5+139.8)  # We substract the mesuring tool
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
            gripper_state['Dx'] = wsg_list[0]['width']/2.0
            gripper_state['Dz'] = 139.8 + 72.5 + 160 # Base + wsg + finger

            gs_point_list = [
                (31.758064516128911, 96.274193548387103),
                (22.725806451612812, 229.17741935483869),
                (193.04838709677409, 136.2741935483871),
                (184.01612903225794, 240.79032258064515),
                (176.27419354838702, 416.27419354838707),
                (182.72580645161281, 532.40322580645159),
                (304.01612903225799, 154.33870967741936),
                (298.85483870967732, 261.43548387096769),
                (292.40322580645153, 405.95161290322585),
                (287.24193548387086, 513.04838709677415),
                (31.758064516128911, 186.59677419354838),
                (21.435483870967573, 305.30645161290317),
                (189.17741935483861, 216.2741935483871),
                (185.30645161290312, 322.08064516129036),
                (180.14516129032251, 496.27419354838707),
                (180.14516129032251, 609.82258064516122),
                (305.30645161290317, 226.59677419354838),
                (294.98387096774184, 327.24193548387098),
                (289.82258064516122, 478.20967741935476),
                (297.5645161290322, 587.88709677419354),
                (113.04838709677409, 205.95161290322582),
                (105.30645161290312, 307.88709677419354),
                (249.82258064516122, 214.98387096774192),
                (242.08064516129025, 320.79032258064512),
                (243.37096774193537, 484.66129032258061),
                (242.08064516129025, 596.91935483870964),
                (349.17741935483866, 226.59677419354838),
                (344.01612903225799, 327.24193548387098),
                (337.5645161290322, 475.62903225806451),
                (338.85483870967732, 573.69354838709671)
            ]

            for j in range(10):
                # print "Touch point: " + str(j+1)
                # if gs_id == 1:
                #     point = self.getCoord(gs1_list[0])
                # elif gs_id == 2:
                #     point = self.getCoord(gs2_list[0])
                point = gs_point_list[j*i+j]
                print point
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
        x0 = (0.1, .0, 0.07, .001, -32.0, 16.0, 13.0)
        bounds = [
            (0, 1),
            (-.5, .5),
            (0, 1),
            (-.5, .5),
            (-50, 50),
            (-50, 50),
            (-50, 50)
        ]
        # res = minimize(eq_sys, x0, bounds=bounds, options={'xtol': 1e-8, 'disp': False})
        res = minimize(eq_sys, x0, method='Nelder-Mead', options={'xtol': 1e-8, 'disp': True})
        print res
        (k1, k2, l1, l2, dx, dy, dz) = res.x

        return [k1, k2, l1, l2, dx, dy, dz]

    def test(self, point, path, num, params):
        cart, gs1_list, gs2_list, wsg_list = self.get_contact_info(path, num)

        gripper_state = {}
        gripper_state['pos'] = cart[0:3]
        gripper_state['quaternion'] = cart[-4:]
        gripper_state['Dx'] = wsg_list[0]['width']/2.0
        gripper_state['Dz'] = 139.8 + 72.5 + 160  # Base + wsg + finger

        point_wb = pxb_2_wb(point, 1, gripper_state, params)
        return point_wb


if __name__ == "__main__":
    pc = PosCalib()
    path = 'pos_calibration/pos_calibration_squares'

    cts = pc.get_px2mm_params()

    #cts = [0.070278739910205668, -2.892043425399342e-08, -0.048533406007308946, -3.1917891601505516e-05, -75.006081343534504, 3.2693008903677541, 15.132586202095485]


    print cts

    point = (0, 320)
    point_wb = pc.test(point, path, 2, cts)
    print point_wb
    dif = point_wb - (487, 605, 386)
    print dif
    print np.linalg.norm(dif)


    point = (242.08064516129025, 320.79032258064512)
    point_wb = pc.test(point, path, 2, cts)
    print point_wb
    dif = point_wb - (485+7.3, 607, 378.3-11.12)
    print dif
    print np.linalg.norm(dif)
