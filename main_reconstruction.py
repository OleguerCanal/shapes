import math, cv2, os, pickle, scipy.io
from processing.location import Location
from processing.raw2pxb import RAW2PXB
from processing.icp import *
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import inv
import pypcd


def cout(cart, wsg):
    x = []
    for i in range(3):
        x.append(100*cart[i])

    print x
    print cart[-4:]
    print wsg['width']

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

def dictPC2npPC(pointcloud):
    mat = []
    for i in range(len(pointcloud['x'])):
        a = np.asarray([pointcloud['x'][i], pointcloud['y'][i], pointcloud['z'][i]])
        mat.append(a)
    return np.asarray(mat)

def npPC2dictPC(np_pointcloud):
    pointcloud = {'x': [], 'y': [], 'z': []}
    for i in range(np_pointcloud.shape[0]):
        pointcloud['x'].append(np_pointcloud[i][0])
        pointcloud['y'].append(np_pointcloud[i][1])
        pointcloud['z'].append(np_pointcloud[i][2])
    return pointcloud

def get_string_pc(pointcloud):
    string = "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH " + str(pointcloud.shape[0]) + "\nHEIGHT 1\nVIEWPOINT 0.0 0.0 0.0 1.0 0.0 0.0 0.0\nPOINTS " + str(pointcloud.shape[0]) + "\nDATA ascii"
    for elem in pointcloud:
        string += '\n' + str(elem[0]) + ' ' + str(elem[1]) + ' ' + str(elem[2])
    return string

if __name__ == "__main__":
    directory = 'datasets/demo_video_1'

    global_pointcloud = None
    for i in range(3):
        exp = str(i+1)
        cart = get_cart(directory + '/p_' + exp + '/cart.npy')
        wsg = load_obj(directory + '/p_' + exp + '/wsg_1.pkl')
        loc = Location()
        local_pointcloud = loc.get_local_pointcloud(
            gs_id=1,
            loc=cart,
            opening=wsg['width'],
            from_heightmap=True,
            directory=directory,
            num=i+1)
        loc.visualize_pointcloud(local_pointcloud)
        local_pointcloud_arr = dictPC2npPC(local_pointcloud)
        # scipy.io.savemat('processing/sample_data/arrdata'+ str(i+2) +'.mat', mdict={'arr': local_pointcloud})

        # Save in local_pointcloud format
        # cloud = pcl.PointCloud()
        # cloud.from_array(local_pointcloud)
        # pcl.save(cloud, 'processing/c++/cloud' + str(i+2) + '.pcd', format = 'pcd')
        # print local_pointcloud
        # pc = pypcd.PointCloud.from_array(local_pointcloud_arr)
        # pc.data = dictPC2npPC(local_pointcloud, form=False)
        # pc.save('processing/c++/cloud' + str(i+2) + '.pcd')
        if global_pointcloud is None:
            global_pointcloud = local_pointcloud
        else:
            global_pointcloud = loc.merge_pointclouds(global_pointcloud, local_pointcloud)

        name = 'processing/c++/cloud' + str(i+5) + '.pcd'
        with open(name, 'w') as f:
            f.write(get_string_pc(local_pointcloud_arr))

    loc.visualize_pointcloud(global_pointcloud)
    # Pyhton built PointCloud
    # global_pointcloud = None
    #
    # for i in range(2):
    #     exp = str(i+1)
    #     cart = get_cart(directory + '/p_' + exp + '/cart.npy')
    #     wsg = load_obj(directory + '/p_' + exp + '/wsg_1.pkl')
    #     loc = Location()
    #     local_pointcloud = loc.get_local_pointcloud(
    #         gs_id=1,
    #         loc=cart,
    #         opening=wsg['width'],
    #         from_heightmap=True,
    #         directory=directory,
    #         num=i+1)
    #     loc.visualize_pointcloud(local_pointcloud)
    #     local_pointcloud = dictPC2npPC(local_pointcloud)
    #     if global_pointcloud is None:
    #         global_pointcloud = local_pointcloud
    #     else:
    #         global_pointcloud, distances, iterations = icp(global_pointcloud, local_pointcloud, tolerance=0.000001)
    #     # global_pointcloud = loc.merge_pointclouds(global_pointcloud, local_pointcloud)
    #     loc.visualize_pointcloud(npPC2dictPC(global_pointcloud))

    # path = 'data_processing/sample_data/'
    # gs_back = cv2.imread(path + 'arc/gs_image.png')
    # gs2_back = cv2.imread(path + 'arc/gs_image2.png')
    #
    # global_pointcloud = {'x': [], 'y': [], 'z': []}
    # for i in range(3):
    #     exp = str(i+1)
    #     gs2 = cv2.imread(path + 'p' + exp + '/gs_image2.png')
    #     cart = get_cart(path + 'p' + exp + '/cart.npy')
    #     wsg = load_obj(path + 'p' + exp + '/wsg.pkl')
    #     loc = Location()
    #     local_pointcloud = loc.get_local_pointcloud(
    #         gs_img=gs2,
    #         gs_back=gs2_back,
    #         gs_id=2,
    #         loc=cart,
    #         opening=wsg['width'])
    #     # loc.visualize_pointcloud(local_pointcloud)
    #     global_pointcloud = loc.merge_pointclouds(global_pointcloud, local_pointcloud)
    #
    # for i in range(3):
    #     exp = str(i+1)
    #     gs = cv2.imread(path + 'p' + exp + '/gs_image.png')
    #     cart = get_cart(path + 'p' + exp + '/cart.npy')
    #     wsg = load_obj(path + 'p' + exp + '/wsg.pkl')
    #     loc = Location()
    #     local_pointcloud = loc.get_local_pointcloud(
    #         gs_img=gs,
    #         gs_back=gs2_back,
    #         gs_id=1,
    #         loc=cart,
    #         opening=wsg['width'])
    #     # loc.visualize_pointcloud(local_pointcloud)
    #     global_pointcloud = loc.merge_pointclouds(global_pointcloud, local_pointcloud)
    #
    # loc.visualize_pointcloud(global_pointcloud)
