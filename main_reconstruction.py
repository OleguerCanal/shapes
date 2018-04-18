import math, cv2, os, pickle, scipy.io, pypcd, subprocess
from processing.location import Location
from processing.raw2pxb import RAW2PXB
# from processing.icp import *
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import inv

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

def load_cpl_pointcloud(path):
    pass

def stitch_pointclouds(fixed, moved):
    # 1. We save the pointclouds in .pcd format
    name = 'processing/c++/cloud' + str(0) + '.pcd'
    with open(name, 'w') as f:
        f.write(get_string_pc(fixed))

    name = 'processing/c++/cloud' + str(1) + '.pcd'
    with open(name, 'w') as f:
        f.write(get_string_pc(moved))

    # 2. We run the c++ program to stitch them
    command = 'cd processing/c++/; ./pairwise_incremental_registration cloud[0-1].pcd'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()

    # 3. We load and return the merged pointcloud
    path = 'processing/c++/1.pcd'
    pc = pypcd.PointCloud.from_path(path)
    # pointcloud = npPC2dictPC(pc.pc_data)
    return pc.pc_data


if __name__ == "__main__":
    directory = 'datasets/stitching_dataset'

    global_pointcloud = None
    processed_global_pointcloud = None
    for i in range(3):
        print "Processing img " + str(i) + "..."
        exp = str(i)
        cart = get_cart(directory + '/p_' + exp + '/cart.npy')
        wsg = load_obj(directory + '/p_' + exp + '/wsg_1.pkl')
        loc = Location()
        local_pointcloud = loc.get_local_pointcloud(
            gs_id=1,
            loc=cart,
            opening=wsg['width'],
            from_heightmap=True,
            directory=directory,
            num=i)

        local_pointcloud_arr = dictPC2npPC(local_pointcloud)

        if global_pointcloud is None:
            global_pointcloud = local_pointcloud                # Dict format
            processed_global_pointcloud = local_pointcloud_arr  # Numpy format
        else:
            global_pointcloud = loc.merge_pointclouds(global_pointcloud, local_pointcloud)
            processed_global_pointcloud = stitch_pointclouds(processed_global_pointcloud, local_pointcloud_arr)

        # print(type(processed_global_pointcloud))
        # loc.visualize_pointcloud(local_pointcloud)
        # loc.visualize_pointcloud(global_pointcloud)
        # loc.visualize_pointcloud(npPC2dictPC(processed_global_pointcloud))

    print "DONE :)"

    loc.visualize_pointcloud(global_pointcloud)
    loc.visualize_pointcloud(npPC2dictPC(processed_global_pointcloud))
