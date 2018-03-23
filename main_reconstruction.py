import math, cv2, os, pickle
from data_processing.location import Location
from data_processing.raw2pxb import RAW2PXB
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

if __name__ == "__main__":
    directory = 'datasets/demo_video'
    global_pointcloud = {'x': [], 'y': [], 'z': []}

    for i in range(3):
        exp = str(i+4)
        cart = get_cart(directory + '/p_' + exp + '/cart.npy')
        wsg = load_obj(directory + '/p_' + exp + '/wsg_1.pkl')
        loc = Location()
        local_pointcloud = loc.get_local_pointcloud(
            gs_id=1,
            loc=cart,
            opening=wsg['width'],
            from_heightmap=True,
            directory=directory,
            num=i+4)
        loc.visualize_pointcloud(local_pointcloud)
        global_pointcloud = loc.merge_pointclouds(global_pointcloud, local_pointcloud)
    loc.visualize_pointcloud(global_pointcloud)
#


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
