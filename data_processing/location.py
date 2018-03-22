from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from raw2pxb import RAW2PXB
from pxb2gsb import PXB2GSB
from gsb2wb import GSB2WB
import numpy as np
from numpy.linalg import inv
import math
import cv2


class Location():
    # This class is to construct pointcloud in world space from:
    # RAW_GS, POS, Gripper opening
    def __init__(self, params={}):
        self.params = params
        if params == {}:
            self.params['origin'] = [0, 0, 0]
            self.params['gripper_point_2_gs_origin'] = 385
            self.params['max_height'] = 8
            self.params['compress_factor'] = 0.1

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
        ax.scatter3D(pointcloud['x'], pointcloud['y'], pointcloud['z'])

        # Set viewpoint.
        ax.azim = 60
        ax.elev = 30

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
        plt.show()

    def translate_pointcloud(self, pointcloud, v):
        for i in range(len(pointcloud['x'])):
            pointcloud['x'][i] += v[0]
            pointcloud['y'][i] += v[1]
            pointcloud['z'][i] += v[2]
        return pointcloud

    def get_local_pointcloud(self, gs_img, gs_back, gs_id, loc, opening):
        # 1. We convert the raw image to pixel_base data
        r2p = RAW2PXB()
        pxb_data = r2p.crop_contact(gs_back, gs_img, gel_id=gs_id, compress_factor=self.params['compress_factor'])
        # r2p.show_image(img=pxb_data)

        self.params['px2mm_params']['img_height'] = pxb_data.shape[0]
        self.params['px2mm_params']['img_width'] = pxb_data.shape[1]

        # 2. We convert pixel_base data into a gs_base pointcloud
        pxb2gsb = PXB2GSB()
        gsb_pointcloud = pxb2gsb.get_gsb_pointcloud(pxb_data, self.params['px2mm_params'], opening, gs_id=gs_id, max_height=3)

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
