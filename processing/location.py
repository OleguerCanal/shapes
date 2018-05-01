from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from raw2pxb import RAW2PXB
import numpy as np
from numpy.linalg import inv
import math, cv2, os, pickle
from world_positioning import pxb_3d_2_wb


class Location():
    def __init__(self):
        self.compress_factor = .2
        self.params_gs1 = [0.0792, -0.0018, -0.0697, -0.0021, 6.4471, 5.9929, 14.3968]
        pass

    def visualize_pointcloud(self, pointcloud):
        ax = plt.axes(projection='3d')
        # ax.scatter3D(pointcloud['x'], pointcloud['y'], pointcloud['z'],
        #     c=pointcloud['x'], cmap='Greens')
        ax.scatter3D(pointcloud['x'], pointcloud['y'], pointcloud['z'],
            c=pointcloud['y'], cmap='Blues')

        # Set viewpoint.
        ax.azim = 135
        ax.elev = 15

        # Label axes.
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')

        def axisEqual3D(ax):
            extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
            sz = extents[:, 1] - extents[:, 0]
            centers = np.mean(extents, axis=1)
            maxsize = max(abs(sz))
            r = maxsize/2
            for ctr, dim in zip(centers, 'xyz'):
                getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

        axisEqual3D(ax)
        plt.gca().invert_xaxis()
        plt.show()

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

        gs1_back = None
        gs2_back = None

        try:
            gs1_back = cv2.imread(directory + '/air/GS1.png')
        except:
            print 'GS1 backgorund not found'
        try:
            gs2_back = cv2.imread(directory + '/air/GS2.png')
        except:
            print 'GS2 backgorund not found'

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

        force_list = []
        for elem in wsg_list:
            force_list.append(elem['force'])

        return cart, gs1_list, gs2_list, wsg_list, force_list, gs1_back, gs2_back

    def get_local_pointcloud(self, gs_id, directory='', num=-1):
        # 1. We convert the raw image to height_map data
        r2p = RAW2PXB()
        cart, gs1_list, gs2_list, wsg_list, force_list, gs1_back, gs2_back = self.get_contact_info(directory, num)
        if gs_id == 1:
            height_map = r2p.multiple_image_processing(
                gel_id=gs_id,
                img_back=gs1_back,
                img_list=gs1_list,
                force_list=force_list,
                compress_factor=self.compress_factor)
            # r2p.show_2images(gs1_list[0][:-80, 40:-50], height_map)
        elif gs_id == 2:
            height_map = r2p.multiple_image_processing(
                gel_id=gs_id,
                img_back=gs2_back,
                img_list=gs2_list,
                force_list=force_list,
                compress_factor=self.compress_factor)
            # r2p.show_image(img=height_map)

        # 2. We convert height_map data into world position
        gripper_state = {}
        gripper_state['pos'] = cart[0:3]
        gripper_state['quaternion'] = cart[-4:]
        gripper_state['Dx'] = wsg_list[0]['width']/2.0
        gripper_state['Dz'] = 139.8 + 72.5 + 160 # Base + wsg + finger

        xs = []
        ys = []
        zs = []

        for i in range(height_map.shape[0]):
            for j in range(height_map.shape[1]):
                if(height_map[i][j] != 0):
                    ii = i*480.0/float(height_map.shape[0])
                    jj = j*640.0/float(height_map.shape[1])
                    world_point = pxb_3d_2_wb(
                        point_3d=(ii, jj, float(3.0*height_map[i][j])/250),
                        gs_id=gs_id,
                        gripper_state = gripper_state,
                        fitting_params = self.params_gs1
                    )
                    xs.append(world_point[0])
                    ys.append(world_point[1])
                    zs.append(world_point[2])

        pointcloud = {}
        pointcloud['x'] = xs
        pointcloud['y'] = ys
        pointcloud['z'] = zs

        return pointcloud

    def merge_pointclouds(self, pointcloud1, pointcloud2):
        pointcloud1['x'] += pointcloud2['x']
        pointcloud1['y'] += pointcloud2['y']
        pointcloud1['z'] += pointcloud2['z']
        return pointcloud1

if __name__ == "__main__":
    gs1 = cv2.imread('sample_data/f1.jpg')
    gs1_back = cv2.imread('sample_data/f1_back.jpg')

    loc = Location()
    local_pointcloud = loc.get_local_pointcloud(
        gs_img=gs1,
        gs_back=gs1_back,
        gs_id=2,
        loc=[50.0, 100.0, -20.0, 0.0, 0.0, -1.0, 0.0],
        opening=20)
    loc.visualize_pointcloud(local_pointcloud)
