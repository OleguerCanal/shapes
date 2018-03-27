import cv2
import matplotlib.pyplot as plt
import numpy as np

class PosCalib():
    def __init__(self):
        self.fname = 'sample_data/pos_calibration/p_0/GS1_0.png'
        self.img = cv2.imread(self.fname)
        self.point = ()

    def getCoord(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(self.img)
        cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        return self.point

    def __onclick__(self,click):
        self.point = (click.xdata,click.ydata)
        return self.point


class DepthCalib():
    def __init__(self):
        pass


if __name__ == "__main__":
    pc = PosCalib()
    print pc.getCoord()
