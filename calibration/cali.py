import cv2, math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize
from world_positioning import pxb_2_wb
import os, pickle

gripper_state = {}
gripper_state['pos'] = [.0, .0, .0]
gripper_state['quaternion'] = (.0, .0, 1, .0)
gripper_state['Dx'] = 10
gripper_state['Dz'] = 200

params = (.5, .0,   2, .0,   0, 0, 0)

point = (30, 320 + 50)

print pxb_2_wb(point=point, gs_id=1, gripper_state=gripper_state, fitting_params=params)


# def get_contact_info(directory, num):
#     def get_cart(path):
#         cart = np.load(path)
#         a = cart[3]
#         cart[3] = cart[6]
#         b = cart[4]
#         cart[4] = a
#         c = cart[5]
#         cart[5] = b
#         cart[6] = c
#         return cart
#
#     def load_obj(path):
#         with open(path, 'rb') as f:
#             return pickle.load(f)
#
#     directory += '/p_' + str(num)
#     file_list = os.listdir(directory)
#
#     cart = get_cart(directory + '/cart.npy')
#     gs1_list = []
#     gs2_list = []
#     wsg_list = []
#
#     for elem in file_list:
#         path = directory + '/' + elem
#         if 'GS1' in elem:
#             gs1_list.append(cv2.imread(path))
#         elif 'GS2' in elem:
#             gs2_list.append(cv2.imread(path))
#         elif 'wsg' in elem:
#             wsg_list.append(load_obj(path))
#
#     return cart, gs1_list, gs2_list, wsg_list
#
#
# path = 'pos_calibration/pos_calibration_squares'
#
# for i in range(6):
#     cart, gs1_list, gs2_list, wsg_list = get_contact_info(path, i)
#
#     gripper_state = {}
#     gripper_state['pos'] = cart[0:3]
#     gripper_state['quaternion'] = cart[-4:]
#     gripper_state['Dx'] = wsg_list[0]['width']/2.0
#     gripper_state['Dz'] = 139.8 + 72.5 + 160  # Base + wsg + finger
#
#     print gripper_state['pos']
#     print gripper_state['quaternion']
#     print gripper_state['Dx']
#
#     params = (.5, .0,   2, .0,   0, 0, 0)
#     point = (30, 320 + 50)
