import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import cv2


class PosCalib():
    def __init__(self):
        pass

    def draw_circle(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img,(x,y),100,(255,0,0),-1)
            self.mouseX, self.mouseY = x,y
            print x, y

    def mark_points(self, img_list, pos_list):
        img = np.zeros((512, 512, 3), np.uint8)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_circle)


class DepthCalib():
    def __init__(self):
        pass


if __name__ == "__main__":
    pc = PosCalib()

    img = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', pc.draw_circle)
    # cv2.imshow('image', img)

    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            print 'a'
            break
        elif k == ord('a'):
            print pc.mouseX, pc.mouseY
