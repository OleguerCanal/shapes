import rospy
import numpy as np

import math
from visualization_msgs.msg import *
from robot_comm.srv import *
from visualization_msgs.msg import *
from wsg_50_common.msg import Status
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import pickle
import time


class DataCollector():
    def __init__():
        self.bridge = CvBridge()

    def __save(self, path, obj, name):
        with open(path + '/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def __callback(self, data, key):
        self.subscribers[key].unregister()
        if topic_dict[key]['msg_format'] == Image:
            try:
                cv2_img = bridge.imgmsg_to_cv2(data, 'rgb8')
            except CvBridgeError, e:
                print(e)
            self.data_recorded[key] = cv2_img
        elif key == 'wsg_driver':
            ws50message = data
            data_dict = {}
            data_dict['status'] = ws50message.status
            data_dict['width'] = ws50message.width
            data_dict['speed'] = ws50message.speed
            data_dict['acc'] = ws50message.acc
            data_dict['force'] = ws50message.force
            data_dict['force_finger0'] = ws50message.force_finger0
            data_dict['force_finger1'] = ws50message.force_finger1
            self.data_recorded[key] = data_dict
        return

    def getCart(self):
        getCartRos_ = rospy.ServiceProxy('/robot1_GetCartesian', robot_GetCartesian)

        try:
            rospy.wait_for_service('/robot1_GetCartesian', timeout = 0.5)
            cart_pose = getCartRos_()
            cart_pose_list = [cart_pose.x/1000.0,cart_pose.y/1000.0,cart_pose.z/1000.0, cart_pose.qx, cart_pose.qy, cart_pose.qz,cart_pose.q0]
            return cart_pose_list
        except:
            print '[Robot] Robot seeems not connected, skipping setCart()'
            return False

    def __save_data(self, get_gs1=False, get_gs2=True, get_wsg=True, directory, iteration):
        if get_wsg is True:
            self.__save(directory, data_recorded['wsg_driver'], 'wsg_'+ str(iteration))
        if get_gs1 is True:
            cv2.imwrite(directory+'/GS1_0' + str(iteration) + '.png', data_recorded['gs_image'])
        if get_gs2 is True:
            cv2.imwrite(directory+'/GS2_0' + str(iteration) + '.png', data_recorded['gs_image2'])


    def get_data(self, get_cart=False, get_gs1=False, get_gs2=True, get_wsg=True, save=True, directory='', iteration=0):
        # 1. We get the cartesian pos from the robot
        if get_cart is True:
            self.cart = getCart()

        # 2. We listen and save the following topics
        self.topic_dict = {
            'wsg_driver': {'topic': '/wsg_50_driver/status', 'msg_format': Status},
            'gs_image': {'topic': 'rpi/gelsight/flip_raw_image', 'msg_format': Image},
            'gs_image2': {'topic': 'rpi/gelsight/flip_raw_image2', 'msg_format': Image}}
        if get_wsg is False:
            self.topic_dict.pop('wsg_driver', None)
        if gs_image is False:
            self.topic_dict.pop('gs_image', None)
        if gs_image2 is False:
            self.topic_dict.pop('gs_image2', None)

        self.data_recorded = {}
        self.subscribers = {}

        rospy.init_node('listener', anonymous=True) # Maybe we should only initialize one general node
        for key in self.topic_dict:
            topic = self.topic_dict[key]['topic']
            msg_format = self.topic_dict[key]['msg_format']
            self.subscribers[key] = rospy.Subscriber(topic, msg_format, __callback, key)
        time.sleep(0.5) # We whait for 0.5 seconds

        # 3. We save things
        if save is True:
            self.__save_data(get_gs1=get_gs1, get_gs2=get_gs2, get_wsg=get_wsg, directory=directory, iteration=iteration)
