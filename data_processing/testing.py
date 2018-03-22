from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from process_gs_images import *


def img2pointCloud(img, normal, gripper_distance, off=(0,0)):
    xs = []
    ys = []
    zs = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j] != 0):
                xs.append(i + off[0])
                ys.append(j + off[1])
                z = normal*(gripper_distance/2+float(img[i][j])/50.0)
                zs.append(z)
    return xs, ys, zs

def get_image(contact_name, back_name, gel_id=1):
    gs1 = cv2.imread(contact_name)
    gs1_back = cv2.imread(back_name)
    no_back = crop_contact(gs1_back, gs1, gel_id=gel_id, is_zeros=True)
    img = cv2.cvtColor(no_back, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
    return img

if __name__ == "__main__":
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    img = get_image(contact_name='f1.jpg', back_name='f1_back.jpg', gel_id=1)
    xdata, ydata, zdata = img2pointCloud(img=img, normal=1, gripper_distance=4)
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

    img = get_image(contact_name='f2.jpg', back_name='f2_back.jpg', gel_id=2)
    xdata, ydata, zdata = img2pointCloud(img=img, normal=-1, gripper_distance=4, off=(40, 0))
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Reds')


    #3. Plot 3d map
    plt.show()
