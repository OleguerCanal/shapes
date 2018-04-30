import cv2, math
import matplotlib.pyplot as plt
import numpy as np
import os, pickle

class fittingPackage():
    def __init__(self):
        pass

    def __getCoord(self, img):
        fig = plt.figure()
        # img = cv2.flip(img, 1)  # We horizontally flip the image
        plt.imshow(img)
        cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()
        return (self.point[1], self.point[0])  # Fix axis values

    def __onclick__(self, click):
        self.point = (click.xdata, click.ydata)
        plt.close('all')
        return self.point

    def __get_contact_info(self, directory, num):
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

    def get_real_point_list(self, name):
        if name == "squares":
            squares_distance = [
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
            for elem in squares_distance:
                x = 480 + elem[1]
                y = 604
                z = 377 - elem[0] # - (250+72.5+139.8)  # We substract the mesuring tool
                point_list.append((x, y, z))

        elif name == "line":
            point_list = []
            for i in range(6):
                point_list.append((1277., 255., 427))

        elif name == "border":
            point_list = []

        return point_list

    def create_package(self, name, gs_id, n_points, from_list=False, from_path='', touches_list=[0], save_path=''):
        real_point_list = get_real_point_list(name=name)

        mc = []
        for i in touches_list:
            # We load the information about that touch
            cart, gs1_list, gs2_list, wsg_list = self.__get_contact_info(from_path, i)

            # We reshape the gripper state information
            gripper_state = {}
            gripper_state['pos'] = cart[0:3]
            gripper_state['quaternion'] = cart[-4:]
            gripper_state['Dx'] = wsg_list[0]['width']/2.0
            gripper_state['Dz'] = 139.8 + 72.5 + 160  # Base + wsg + finger

            for j in range(n_points):
                print "Touch point: " + str(j)
                if gs_id == 1:
                    gs_point = self.getCoord(gs1_list[0])
                elif gs_id == 2:
                    gs_point = self.getCoord(gs2_list[0])
                real_point = real_point_list(j)

                print "Matched: " + str(gs_point) + " with " + str(real_point)

                mc.append((gs_point, gs_id, gripper_state, real_point))

        np.save(path + '/' + name + '.npy', mc)


if __name__ == "__main__":
    name = 'squares'
    n_points = 12
    from_path = 'pos_calibration/pos_calibration_squares'
    touches_list = range(7)
    save_path = 'pos_calibration'
    gs_id = 1

    fp = fittingPackage()
    fp.create_package(
        name=name,
        gs_id=gs_id,
        n_points = n_points
        from_list=False,
        from_path=from_path,
        touches_list=touches_list,
        saving_path=save_path
    )
