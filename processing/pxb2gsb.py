from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from raw2pxb import RAW2PXB
import numpy as np
import cv2


class PXB2GSB():
    # This class is to construct pointcloud in gs_base from pixel_base data
    # pixel_base data is a grayscale image where each pixel shows height
    def __init__(self):
        pass

    # This function is VEEEERY improvable
    def __pixel2mm(self, x, y, params):
        x = x*params['gs_height']/float(params['img_height'])
        y = y*params['gs_width']/float(params['img_width'])
        return x, y
        # k1, k2, l1, l2 = [0.073823915173035479, 9.926577535826757e-06, 0.060562277447977549, -4.4780153261919538e-06]
        # fx = x*(k1 + k2*y)
        # fy = y*(l1 + l2*x)
        # return (fx, fy)

    # This function is also VEEEERY improvable
    def get_gsb_pointcloud(self, pxb_data, px2mm_params, opening, gs_id=1, max_height=3):
        if gs_id == 1:
            normal = 1
        else:
            normal = -1

        xs = []
        ys = []
        zs = []
        for i in range(pxb_data.shape[0]):
            for j in range(pxb_data.shape[1]):
                if(pxb_data[i][j] != 0):
                    x, y = self.__pixel2mm(i, j, px2mm_params)
                    z = normal*(opening+float(max_height*pxb_data[i][j])/255.0)
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)

        pointcloud = {}
        pointcloud['x'] = xs
        pointcloud['y'] = ys
        pointcloud['z'] = zs
        return pointcloud

    def visualize_pointcloud(self, pointcloud):
        ax = plt.axes(projection='3d')
        ax.scatter3D(pointcloud['x'], pointcloud['y'], pointcloud['z'],
            c=pointcloud['z'], cmap='Greens')
        plt.show()


if __name__ == "__main__":
    gs1 = cv2.imread('sample_data/f1.jpg')
    gs1_back = cv2.imread('sample_data/f1_back.jpg')

    # 1. We convert the raw image to pixel_base data
    r2p = RAW2PXB()
    pxb_data = r2p.crop_contact(gs1_back, gs1, gel_id=2, compress_factor=0.2)
    # r2p.show_image(img=pxb_data)

    # 2. We convert pixel_base data into a gs_base pointcloud
    # Pixel to mm mapping params
    px2mm_params = {}
    px2mm_params['gs_height'] = 45.0
    px2mm_params['gs_width'] = 45.0
    px2mm_params['img_height'] = pxb_data.shape[0]
    px2mm_params['img_width'] = pxb_data.shape[1]

    pxb2gsb = PXB2GSB()
    gsb_data = pxb2gsb.get_gsb_pointcloud(pxb_data, px2mm_params, gs_id=2, max_height=3)
    gsb_data = pxb2gsb.visualize_pointcloud(gsb_data)
