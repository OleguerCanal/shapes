from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import math
import cv2

def __quaternion_matrix(quaternion):
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    _EPS = 1e-5
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def __grb2wb(point, gripper_pos, quaternion):
    w2gr_mat = __quaternion_matrix(quaternion)
    v = (point[0], point[1], point[2], 1.0)
    #print "v in gripper base: " + str(v)
    v = w2gr_mat.dot(v)
    #print "v in world base: " + str(v)
    #print "Gripper pos: " + str(gripper_pos*1000)

    # for i in range(3):
    #     gripper_pos[i] = gripper_pos[i]*1000
    # return v[0:3] + gripper_pos
    return v[0:3] + 1000*gripper_pos

def pxb_2_wb(point, gs_id, gripper_state, fitting_params):
    x, y = point
    if gs_id == 1:
        normal = 1
    else:
        normal = -1

    pos = gripper_state['pos']
    quaternion = gripper_state['quaternion']
    Dx = gripper_state['Dx'] # Obertura
    Dz = gripper_state['Dz']

    # k0, k1, k2, l0, l1, l2, dx, dy, dz = fitting_params
    k1, k2,  l1, l2,  dx, dy, dz = fitting_params
    # k1, k2,  l1, l2 = 7.79474020e-02, -1.69925499e-03, -6.67926496e-02, 7.24325417e-04
    # dx, dy, dz = fitting_params

    p1 = (x, y - 640.0/2)
    # p2 = (p1[0]*k1 + p1[1]*k2 + k0*p1[0]*p1[1], p1[1]*l1 + p1[0]*l2 + l0*p1[0]*p1[1])
    p2 = (p1[0]*k1 + p1[1]*k2, p1[1]*l1 + p1[0]*l2)
    p3 = (normal*(Dx + dx), p2[1] + dy, Dz + dz + p2[0])
    p4 = __grb2wb(point=p3, gripper_pos=pos, quaternion=quaternion)
    # print "p1: " + str(p1)
    # print "p2: " + str(p2)
    # print "p3: " + str(p3)
    # print "p4: " + str(p4)
    return p4

def pxb_3d_2_wb(point_3d, gs_id, gripper_state, fitting_params):
    x, y, z = point_3d
    if gs_id == 1:
        normal = 1
    else:
        normal = -1

    pos = gripper_state['pos']
    quaternion = gripper_state['quaternion']
    Dx = gripper_state['Dx'] # Obertura
    Dz = gripper_state['Dz']

    # k0, k1, k2, l0, l1, l2, dx, dy, dz = fitting_params
    k1, k2,  l1, l2,  dx, dy, dz = fitting_params
    # k1, k2,  l1, l2 = 7.79474020e-02, -1.69925499e-03, -6.67926496e-02, 7.24325417e-04
    # dx, dy, dz = fitting_params

    p1 = (x, y - 640.0/2, z)
    # p2 = (p1[0]*k1 + p1[1]*k2 + k0*p1[0]*p1[1], p1[1]*l1 + p1[0]*l2 + l0*p1[0]*p1[1])
    p2 = (p1[0]*k1 + p1[1]*k2, p1[1]*l1 + p1[0]*l2, p1[2])
    p3 = (normal*(Dx + dx + p2[2]), p2[1] + dy, Dz + dz + p2[0])
    p4 = __grb2wb(point=p3, gripper_pos=pos, quaternion=quaternion)
    # print "p1: " + str(p1)
    # print "p2: " + str(p2)
    # print "p3: " + str(p3)
    # print "p4: " + str(p4)
    return p4
