from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from raw2pxb import RAW2PXB
from pxb2gsb import PXB2GSB
import numpy as np
from numpy.linalg import inv
import math
import cv2


class GSB2WB():
    # This class is to convert a gs_base pointcloud to a gr_base pointcloud
    # Merging this class with the previous would be much more time efficient
    def __init__(self):
        self._EPS = 1e-5
        pass

    def __gsb2grb(self, pointcloud, gripper_point_2_gs_origin):
        for i in range(len(pointcloud['x'])):
            x = -pointcloud['z'][i]
            y = pointcloud['y'][i]
            z = pointcloud['x'][i]
            pointcloud['x'][i] = float(x)
            pointcloud['y'][i] = float(y)
            pointcloud['z'][i] = float(z + gripper_point_2_gs_origin)
            # We added displacement from the center of the griper base to the gs_base
        return pointcloud

    def __quaternion_matrix(self, quaternion):
        q = np.array(quaternion, dtype=np.float64, copy=True)
        n = np.dot(q, q)
        if n < self._EPS:
            return np.identity(4)
        q *= math.sqrt(2.0 / n)
        q = np.outer(q, q)
        return np.array([
            [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
            [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
            [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
            [                0.0,                 0.0,                 0.0, 1.0]])

    def __grb2wb(self, pointcloud, quaternion):
        # print quaternion
        w2gr_mat = self.__quaternion_matrix(quaternion)
        # print "Rotation Matrix:"
        # print w2gr_mat
        for i in range(len(pointcloud['x'])):
            v = [pointcloud['x'][i], pointcloud['y'][i], pointcloud['z'][i], 1.0]
            # print v
            # w2gr_mat = w2gr_mat.transpose()
            # v = np.linalg.solve(w2gr_mat, v)
            # v = w2gr_mat.dot(v)
            # w2gr_mat = w2gr_mat.transpose()
            v = w2gr_mat.dot(v)

            pointcloud['x'][i] = v[0]
            pointcloud['y'][i] = v[1]
            pointcloud['z'][i] = v[2]
        return pointcloud

    # def __gsb2wb(self, pointcloud, quaternion, gripper_point_2_gs_origin):
    #     # One-step gsb2wb (more time efficient)
    #     w2gr_mat = self.__quaternion_matrix(quaternion)
    #     for i in range(len(pointcloud['x'])):
    #         v = [-pointcloud['z'][i], pointcloud['y'][i], float(pointcloud['x'][i]+gripper_point_2_gs_origin), 1]
    #         v = w2gr_mat.dot(v)
    #         pointcloud['x'][i] = v[0]
    #         pointcloud['y'][i] = v[1]
    #         pointcloud['z'][i] = v[2]
    #     return pointcloud

    def get_wb_pointcloud(self, pointcloud, quaternion, gripper_point_2_gs_origin=0):
        # print quaternion
        grb_pointcloud = self.__gsb2grb(pointcloud, gripper_point_2_gs_origin)
        # self.visualize_pointcloud(grb_pointcloud)
        # grb_pointcloud = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
        # print quaternion
        # print '******'
        # self.print_points(grb_pointcloud)
        wb_pointcloud = self.__grb2wb(grb_pointcloud, quaternion)
        # print '******'
        # self.print_points(wb_pointcloud)
        # self.visualize_pointcloud(wb_pointcloud)
        # wb_pointcloud = self.__gsb2wb(pointcloud, quaternion, gripper_point_2_gs_origin)
        return wb_pointcloud

    def visualize_pointcloud(self, pointcloud):
        ax = plt.axes(projection='3d')
        ax.scatter3D(pointcloud['x'], pointcloud['y'], pointcloud['z'],
            c=pointcloud['x'], cmap='Greens')
        plt.show()

    def print_points(self, pointcloud):
        for i in range(len(pointcloud['x'])):
            print str(pointcloud['x'][i]) + ', ' + str(pointcloud['y'][i]) + ', ' + str(pointcloud['z'][i])

if __name__ == "__main__":
    # gs1 = cv2.imread('sample_data/f1.jpg')
    # gs1_back = cv2.imread('sample_data/f1_back.jpg')
    #
    # # 1. We convert the raw image to pixel_base data
    # r2p = RAW2PXB()
    # pxb_data = r2p.crop_contact(gs1_back, gs1, gel_id=2, compress_factor=0.2)
    # # r2p.show_image(img=pxb_data)
    #
    # # 2. We convert pixel_base data into a gs_base pointcloud
    # # Pixel to mm mapping params
    # px2mm_params = {}
    # px2mm_params['gs_height'] = 45.0
    # px2mm_params['gs_width'] = 45.0
    # px2mm_params['img_height'] = pxb_data.shape[0]
    # px2mm_params['img_width'] = pxb_data.shape[1]
    #
    # pxb2gsb = PXB2GSB()
    # gsb_pointcloud = pxb2gsb.get_gsb_pointcloud(pxb_data, px2mm_params, opening=50, gs_id=2, max_height=3)
    # # gsb_pointcloud = pxb2gsb.visualize_pointcloud(gsb_pointcloud)

    # 3. We convert gs_base pointcloud to world_base pointcloud
    # Test Values
    gsb_pointcloud = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
    quaternion = [0.0, 0.0, -1.0, 0.0]
    # quaternion = [0.0, -7.08200000e-01, -7.06000000e-01, 0.0]
    # quaternion = [0.0, -0.92866, -0.37092, 0.0]
    # quaternion = [0, -0.71628, -0.69782, 0]
    # quaternion = [0.22115, -0.0, -0.97518, 0.0]
    # quaternion = [0.20322, 0.36173, -0.90567, -0.0873]

    gsb2wb = GSB2WB()
    wb_pointcloud = gsb2wb.get_wb_pointcloud(gsb_pointcloud, quaternion)
    # gsb2wb.print_points(wb_pointcloud)
    # gsb2wb.visualize_pointcloud(wb_pointcloud)
