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
        self.point_list = self.__get_point_list()
        self.gs_point_list = [
            # SQUARES
            # (31.758064516128911, 96.274193548387103),
            # (22.725806451612812, 229.17741935483869),
            # (193.04838709677409, 136.2741935483871),
            # (184.01612903225794, 240.79032258064515),
            # (176.27419354838702, 416.27419354838707),
            # (182.72580645161281, 532.40322580645159),
            # (304.01612903225799, 154.33870967741936),
            # (298.85483870967732, 261.43548387096769),
            # (292.40322580645153, 405.95161290322585),
            # (287.24193548387086, 513.04838709677415),
            # (31.758064516128911, 186.59677419354838),
            # (21.435483870967573, 305.30645161290317),
            # (189.17741935483861, 216.2741935483871),
            # (185.30645161290312, 322.08064516129036),
            # (180.14516129032251, 496.27419354838707),
            # (180.14516129032251, 609.82258064516122),
            # (305.30645161290317, 226.59677419354838),
            # (294.98387096774184, 327.24193548387098),
            # (289.82258064516122, 478.20967741935476),
            # (297.5645161290322, 587.88709677419354),
            # (113.04838709677409, 205.95161290322582),
            # (105.30645161290312, 307.88709677419354),
            # (249.82258064516122, 214.98387096774192),
            # (242.08064516129025, 320.79032258064512),
            # (243.37096774193537, 484.66129032258061),
            # (242.08064516129025, 596.91935483870964),
            # (349.17741935483866, 226.59677419354838),
            # (344.01612903225799, 327.24193548387098),
            # (337.5645161290322, 475.62903225806451),
            # (338.85483870967732, 573.69354838709671),
            # SQUARES
            (25.306451612903061, 434.33870967741927),
            (30.467741935483787, 537.5645161290322),
            (176.27419354838702, 107.88709677419354),
            (176.27419354838702, 222.7258064516129),
            (185.30645161290312, 398.20967741935488),
            (194.33870967741927, 502.72580645161281),
            (288.53225806451604, 129.82258064516128),
            (292.40322580645153, 229.17741935483869),
            (296.27419354838702, 378.85483870967744),
            (305.30645161290317, 483.37096774193549),

            (22.725806451612812, 336.27419354838707),
            (29.177419354838548, 453.69354838709671),
            (178.85483870967732, 34.338709677419359),
            (178.85483870967732, 145.30645161290323),
            (182.72580645161281, 318.20967741935488),
            (190.46774193548379, 433.04838709677415),
            (292.40322580645153, 49.822580645161281),
            (287.24193548387086, 158.20967741935485),
            (294.98387096774184, 311.75806451612902),
            (306.5967741935483, 413.69354838709671),

            (125.95161290322568, 337.5645161290322),
            (113.04838709677409, 453.69354838709671),
            (240.79032258064507, 54.98387096774195),
            (242.08064516129025, 158.20967741935485),
            (245.95161290322574, 318.20967741935488),
            (251.11290322580635, 424.01612903225805),
            (340.14516129032251, 73.048387096774206),
            (338.85483870967732, 167.24193548387095),
            (344.01612903225799, 310.4677419354839),
            (350.46774193548379, 404.66129032258061),

            # LINE
            (331.11290322580635, 243.37096774193546),
            (327.24193548387086, 371.11290322580646),
            (327.24193548387086, 507.88709677419354),
            (382.72580645161281, 496.27419354838707),
            (368.53225806451604, 368.5322580645161),
            (369.82258064516122, 243.37096774193546)]

    def getCoord(self, img):
        fig = plt.figure()
        img = cv2.flip(img, 1)
        plt.imshow(img)
        cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()
        return (self.point[1], self.point[0])  # Fix axis values

    def __onclick__(self, click):
        self.point = (click.xdata, click.ydata)
        plt.close('all')
        return self.point

    def __get_point_list(self):
        image_list = [
            (.0, 18.5),
            (.0, 25.38),
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

        for i in range(6):
            # point_list.append((1169.7, 254.69, 427.71))
            dx = 1 + 109.0/2
            dz = 139.8 + 72.5 + 160
            point_list.append((1233.21+dx, 254.69, 923.12-dz - 33.11))

        return point_list

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

        return cart, gs1_list, gs2_list, wsg_list

    def __gather_points_pairs(self):
        self.mc = []

        # path = 'pos_calibration/pos_calibration_squares'
        # gs_id = 1
        # for i in range(3):
        #     cart, gs1_list, gs2_list, wsg_list = self.get_contact_info(path, i)
        #
        #     gripper_state = {}
        #     gripper_state['pos'] = cart[0:3]
        #     gripper_state['quaternion'] = cart[-4:]
        #     gripper_state['Dx'] = wsg_list[0]['width']/2.0
        #     gripper_state['Dz'] = 139.8 + 72.5 + 160  # Base + wsg + finger
        #
        #     print gripper_state
        #
        #     for j in range(10):
        #         # print "Touch point: " + str(j+1)
        #         # if gs_id == 1:
        #         #     point = self.getCoord(gs1_list[0])
        #         # elif gs_id == 2:
        #         #     point = self.getCoord(gs2_list[0])
        #         point = self.gs_point_list[10*i+j]
        #         # print point
        #         self.mc.append((point, gs_id, gripper_state, self.point_list[j][0], self.point_list[j][1], self.point_list[j][2]))

        path = 'pos_calibration/pos_calibration_line'
        gs_id = 1
        for i in range(6):
            cart, gs1_list, gs2_list, wsg_list = self.get_contact_info(path, i)

            gripper_state = {}
            gripper_state['pos'] = cart[0:3]
            gripper_state['quaternion'] = cart[-4:]
            gripper_state['Dx'] = wsg_list[0]['width']/2.0
            gripper_state['Dz'] = 139.8 + 72.5 + 160 # Base + wsg + finger

            print gripper_state

            for j in range(1):
                print "Touch point: " + str(j+1)
                if gs_id == 1:
                    point = self.getCoord(gs1_list[0])
                elif gs_id == 2:
                    point = self.getCoord(gs2_list[0])
                print str(point) + ","
                # point = self.gs_point_list[30+i]
                self.mc.append((point, gs_id, gripper_state, self.point_list[10+i][0], self.point_list[10+i][1], self.point_list[10+i][2]))

        return self.mc

    def get_px2mm_params(self):

        self.mc = self.__gather_points_pairs()
        # for elem in mc:
        #     print elem

        def eq_sys(params):
            dif = 0
            n = len(self.mc)
            self.difs = []
            for (point, gs_id, gripper_state, fx, fy, fz) in self.mc:
                (fp_x, fp_y, fp_z) = pxb_2_wb(point, gs_id, gripper_state, params)
                dif += np.linalg.norm((fp_x-fx, fp_y-fy, fp_z-fz))
                self.difs.append((fp_x-fx, fp_y-fy, fp_z-fz))
            return dif/n  # This number is the average distance between estimations and real points

        # As optimization problem
        # x0 = (.0, 0.1, .0,   .0, 0.07, .0,   1.4, -.24, -1.2)
        x0 = (0.1, .0,  0.065, .0,   2.5, -1.35, -0.25)
        # x0 = (2.0, 0.2, 13.0)
        # bounds = [
        #     (-2, 2),
        #     (0, 1),
        #     (-.5, .5),
        #
        #     (-2, 2),
        #     (0, 1),
        #     (-.5, .5),
        #
        #     (-50, 50),
        #     (-50, 50),
        #     (-50, 50)
        # ]
        # res = minimize(eq_sys, x0, bounds=bounds, options={'xtol': 1e-8, 'disp': False})
        res = minimize(eq_sys, x0, method='Nelder-Mead', options={'xtol': 1e-8, 'disp': True})
        print "Success: " + str(res.success)
        print "Average distance: " + str(res.fun)
        # (k0, k1, k2, l0, l1, l2, dx, dy, dz) = res.x

        # return [k0, k1, k2, l0, l1, l2, dx, dy, dz]
        return res.x

    def test(self, point, path, num, params):
        cart, gs1_list, gs2_list, wsg_list = self.get_contact_info(path, num)

        gripper_state = {}
        gripper_state['pos'] = cart[0:3]
        gripper_state['quaternion'] = cart[-4:]
        gripper_state['Dx'] = wsg_list[0]['width']/2.0
        gripper_state['Dz'] = 139.8 + 72.5 + 160  # Base + wsg + finger

        point_wb = pxb_2_wb(point, 1, gripper_state, params)
        return point_wb

    def test_all(self, params):
        i = 0
        for (point, gs_id, gripper_state, fx, fy, fz) in self.mc:
            i += 1
            (fp_x, fp_y, fp_z) = pxb_2_wb(point, gs_id, gripper_state, params)
            print "Real: " + str((fx, fy, fz))
            print "Guessed: " + str((fp_x, fp_y, fp_z))
            print "Diference: " + str((fp_x-fx, fp_y-fy, fp_z-fz))
            print "Distance: " + str(np.linalg.norm((fp_x-fx, fp_y-fy, fp_z-fz)))
            if i%10 == 0:
                print "*****"

    def test_new_squares(self, params):
        mc_test = []

        test_gs_point_list = [
            (102.72580645161281, 409.82258064516122),
            (114.33870967741922, 522.08064516129025),
            (243.37096774193537, 125.95161290322579),
            (239.49999999999989, 227.88709677419357),
            (242.08064516129025, 387.88709677419354),
            (248.53225806451604, 500.14516129032256),
            (333.69354838709671, 137.56451612903226),
            (337.5645161290322, 235.62903225806454),
            (340.14516129032251, 371.11290322580646),
            (353.04838709677415, 475.62903225806451)
        ]

        path = 'pos_calibration/pos_calibration_squares'
        gs_id = 1
        i = 3
        cart, gs1_list, gs2_list, wsg_list = self.get_contact_info(path, i)

        gripper_state = {}
        gripper_state['pos'] = cart[0:3]
        gripper_state['quaternion'] = cart[-4:]
        gripper_state['Dx'] = wsg_list[0]['width']/2.0
        gripper_state['Dz'] = 139.8 + 72.5 + 160  # Base + wsg + finger

        for j in range(10):
            # if gs_id == 1:
            #     point = self.getCoord(gs1_list[0])
            # elif gs_id == 2:
            #     point = self.getCoord(gs2_list[0])
            # print point
            point = test_gs_point_list[j]
            mc_test.append((point, gs_id, gripper_state, self.point_list[j][0], self.point_list[j][1], self.point_list[j][2]))

        i = 0
        for (point, gs_id, gripper_state, fx, fy, fz) in mc_test:
            i += 1
            (fp_x, fp_y, fp_z) = pxb_2_wb(point, gs_id, gripper_state, params)
            print "Real: " + str((fx, fy, fz))
            print "Guessed: " + str((fp_x, fp_y, fp_z))
            print "Diference: " + str((fp_x-fx, fp_y-fy, fp_z-fz))
            print "Distance: " + str(np.linalg.norm((fp_x-fx, fp_y-fy, fp_z-fz)))
            if i%10 == 0:
                print "*****"




if __name__ == "__main__":
    pc = PosCalib()
    path = 'pos_calibration/pos_calibration_squares'

    cts = pc.get_px2mm_params()

    # cts = [-5.31736713e-04, 1.32689988e-01, 1.92324850e-01, 4.93343916e-04, -2.60680326e-01,  2.58946303e-02,   1.18242193e+02, 2.69041742e-01, -1.72261820e+00]

    print cts
    pc.test_all(params=cts)
    # pc.test_new_squares(params=cts)
