from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from raw2pxb import RAW2PXB
from pxb2gsb import PXB2GSB
from gsb2wb import GSB2WB
import numpy as np
from numpy.linalg import inv
import math, cv2, os, pickle


class Location():
    # This class is to construct pointcloud in world space from:
    # RAW_GS, POS, Gripper opening
    def __init__(self, params={}):
        self.params = params
        if params == {}:
            self.params['origin'] = [0, 0, 0]
            self.params['gripper_point_2_gs_origin'] = 385
            self.params['max_height'] = 10
            self.params['compress_factor'] = 0.2

            px2mm_params = {}
            px2mm_params['gs_height'] = 45.0
            px2mm_params['gs_width'] = 45.0
            self.params['px2mm_params'] = px2mm_params

    def set_origin(self, origin):
        self.params['origin'] = origin

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

    def translate_pointcloud(self, pointcloud, v):
        for i in range(len(pointcloud['x'])):
            pointcloud['x'][i] += v[0]
            pointcloud['y'][i] += v[1]
            pointcloud['z'][i] += v[2]
        return pointcloud

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

    def get_local_pointcloud(self, gs_id, loc, opening, gs_img=None, gs_back=None, from_heightmap=False, directory='', num=-1):
        # 1. We convert the raw image to pixel_base data
        r2p = RAW2PXB()
        if from_heightmap is False:
            height_map = r2p.crop_contact(gs_back, gs_img, gel_id=gs_id, compress_factor=self.params['compress_factor'])
        else:
            cart, gs1_list, gs2_list, wsg_list, force_list, gs1_back, gs2_back = self.get_contact_info(directory, num)
            if gs_id == 1:
                height_map = r2p.multiple_image_processing(
                    gel_id=gs_id,
                    img_back=gs1_back,
                    img_list=gs1_list,
                    force_list=force_list)
                r2p.show_2images(gs1_list[0][:-80, 40:-50], height_map)
            elif gs_id == 2:
                height_map = r2p.multiple_image_processing(
                    gel_id=gs_id,
                    img_back=gs2_back,
                    img_list=gs2_list,
                    force_list=force_list)

        # r2p.show_image(img=height_map)

        self.params['px2mm_params']['img_height'] = height_map.shape[0]
        self.params['px2mm_params']['img_width'] = height_map.shape[1]

        # 2. We convert pixel_base data into a gs_base pointcloud
        pxb2gsb = PXB2GSB()
        gsb_pointcloud = pxb2gsb.get_gsb_pointcloud(height_map, self.params['px2mm_params'], opening, gs_id=gs_id, max_height=3)


        # 3. We convert gs_base pointcloud to world_base pointcloud
        quaternion = loc[-4:]
        gsb2wb = GSB2WB()
        wb_pointcloud = gsb2wb.get_wb_pointcloud(gsb_pointcloud, quaternion, self.params['gripper_point_2_gs_origin'])

        # 4. We apply the translation with reference to the origin
        v = [0, 0, 0]
        for i in range(3):
            v[i] = 1000*loc[i] - self.params['origin'][i]
        local_pointcloud = self.translate_pointcloud(wb_pointcloud, v)
        return local_pointcloud

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
