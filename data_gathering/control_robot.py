from data_collector import DataCollector
from gripper import *
from visualization_msgs.msg import *
from robot_comm.srv import *
from visualization_msgs.msg import *
from wsg_50_common.msg import Status
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge, CvBridgeError
import rospy, math, cv2, os, pickle
import numpy as np
import time

class ControlRobot():
    def __init__(self):
        pass

    def move_cart_mm(self, dx=0, dy=0, dz=0):
        #Define ros services
        getCartRos = rospy.ServiceProxy('/robot1_GetCartesian', robot_GetCartesian)
        setCartRos = rospy.ServiceProxy('/robot1_SetCartesian', robot_SetCartesian)
        #read current robot pose
        c = getCartRos()
        #move robot to new pose
        setCartRos(c.x+dx, c.y+dy, c.z+dz, c.q0, c.qx, c.qy, c.qz)

    def close_gripper_f(self, grasp_speed=50, grasp_force=40):
        graspinGripper(grasp_speed=grasp_speed, grasp_force=grasp_force)

    def open_gripper(self):
        open(speed=100)

    def palpate(self, speed=40, force_list=[3, 6, 10], save=False, path=''):
        dc = DataCollector

        # 0. We create the directory
        if not os.path.exists(path): # If the directory does not exist, we create it
            os.makedirs(path)

        # 1. We get and save the cartesian coord.
        cart = dc.getCart()
        np.save(path + 'cart.npy', cart)

        # 2. We get wsg forces and gs images at every set force and store them
        i = 0
        for force in force_list:
            self.close_gripper_f(grasp_speed=speed, grasp_force=force_list.pop(0))
            if i == 0:
                time.delay(3) # More distance to cover
            else:
                time.delay(1)
            dc.get_data(get_cart=False, get_gs1=False, get_gs2=True, get_wsg=True, save=True, directory=path, iteration=i)
            dc.save_data()
            i += 1

        def perfrom_experiment(self, experiment_name='test', movement_list=[]):
            i = 0
            for movement in movement_list:
                pat = experiment_name + '/p_' + str(i) + '/'
                self.palpate(speed=40, force_list=[3, 6, 10], save=True, path=path)
                self.move_cart_mm(movement[0], movement[1], movement[2])
                time.sleep(5)
