import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from labelling import Labeller

_EPS = 1e-5

average_light = []

def plot(f):
    im = plt.imshow(f, cmap='hot')
    plt.colorbar(im, orientation='horizontal')
    plt.show()
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow("im2", f)
    # cv2.waitKey(0)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def make_kernal(n):
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
    return kernal

def calibration(img,background):
    M = np.load('warp_matrix.npy')
    rows,cols,cha = img.shape
    imgw = cv2.warpPerspective(img, M, (cols, rows))
    imgwc = imgw[12:,71:571,:]
    bg_imgw = cv2.warpPerspective(background, M, (cols, rows))
    bg_imgwc = bg_imgw[12:,71:571,:]
    img_blur = cv2.GaussianBlur(bg_imgwc.astype(np.float32),(25,25),30)
    img_bs = imgwc.astype(np.int32) - img_blur.astype(np.int32) + np.mean(img_blur)
    return img_bs.astype(np.uint8),imgwc

def contact_detection(im,im_ref,low_bar,high_bar):
    im_sub = im/im_ref*70
    im_gray = rgb2gray(im_sub).astype(np.uint8)
    im_canny = cv2.Canny(im_gray,low_bar,high_bar)
    kernal1 = make_kernal(10)
    kernal2 = make_kernal(10)
    kernal3 = make_kernal(30)
    kernal4 = make_kernal(20)
    img_d = cv2.dilate(im_canny, kernal1, iterations=1)
    img_e = cv2.erode(img_d, kernal1, iterations=1)
    img_ee = cv2.erode(img_e, kernal2, iterations=1)
    #    plt.figure()
    #    plt.imshow(img_ee)
    #    plt.show()
    img_dd = cv2.dilate(img_ee, kernal3, iterations=1)
    img_eee = cv2.erode(img_dd, kernal4, iterations=1)
    img_label = np.stack((np.zeros(img_dd.shape),img_eee,np.zeros(img_dd.shape)),axis = 2).astype(np.uint8)
    return img_label

def creat_mask(im_gray):
    mask = (im_gray > 100).astype(np.uint8)
    kernal1 = make_kernal(10)
    kernal2 = make_kernal(30)
    kernal3 = make_kernal(100)
    kernal4 = make_kernal(80)
    img_d = cv2.dilate(mask, kernal1, iterations=1)
    img_e = cv2.erode(img_d, kernal2, iterations=1)
    img_dd = cv2.dilate(img_e, kernal3, iterations=1)
    img_ee = cv2.erode(img_dd, kernal4, iterations=1)
    return img_ee

def is_circle_inside(center, radius, image):
    n_x, n_y = (image.shape[0], image.shape[1])
    c_y, c_x = center

    # We check if part of the circle is out of bounds
    if c_x < radius or c_x + radius > n_x:
        return False
    if c_y < radius or c_y + radius > n_y:
        return False
    return True

def validate_result(center, radius, image, ring_radius_factor=1.65, max_accepted_normalized_light=0.2):
    n_x, n_y = (image.shape[0], image.shape[1])
    c_y, c_x = center
    center = c_y, c_x

    # We check if part of the circle is out of bounds
    if c_x < radius or c_x + radius > n_x:
        return False
    if c_y < radius or c_y + radius > n_y:
        return False

    # We check if there is too much light outside the circle
    xvalues = np.array(range(n_y))
    yvalues = np.array(range(n_x))
    x_pixel, y_pixel = np.meshgrid(xvalues, yvalues)

    x_dif = (x_pixel - center[0]).astype(np.float32)
    y_dif = (y_pixel - center[1]).astype(np.float32)

    original_mask = ((x_dif**2 + y_dif**2) < radius**2).astype(np.float32)
    outside_mask = ((x_dif**2 + y_dif**2) > (radius-4)**2).astype(np.float32)
    ring_mask = ((x_dif**2 + y_dif**2) < (ring_radius_factor*radius)**2).astype(np.float32)
    ring_mask = ring_mask*outside_mask

    grayscale_img = rgb2gray(image)

    # plot(grayscale_img)
    # plot(ring_mask)

    inside_average_light = np.sum((grayscale_img*original_mask).astype(np.float32))/np.sum(original_mask)
    outside_average_light = np.sum((grayscale_img*outside_mask).astype(np.float32))/np.sum(outside_mask)
    ring_average_light = np.sum((grayscale_img*ring_mask).astype(np.float32))/np.sum(ring_mask)

    normalized_light = (ring_average_light-outside_average_light)/(inside_average_light-outside_average_light)
    average_light.append(normalized_light)

    # print normalized_light

    if normalized_light > max_accepted_normalized_light:
        return False
    return True


if __name__ == "__main__":
    # root, dirs, files = os.walk("data/raw2").next()
    input_path = "data/processed2/input/"
    label_path = "data/processed2/labels/"

    ref = cv2.imread('data/raw2/'+'GS2_1.png')
    ref_bs, ref_warp = calibration(ref, ref)

    good_data_list = np.load("data/processed2/good_data_list.npy").tolist()
    bad_data_list = np.load("data/processed2/bad_data_list.npy").tolist()

    # iteration = 0

    paths = np.load("data/processed2/files_name_list.npy")
    print paths
    end = paths.size
    half = end/2 + len(good_data_list) + len(bad_data_list)
    print half
    print end
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    ok_key = int(cv2.waitKey(0))

    iteration = 11102
    for it in range(half, end):
        print "You have completed: " + str(100*float(it-half)/float(end-half)) + "%"
        file_path = paths[it]
        file_name = 'data/raw2/' + file_path
        # if (file_name not in good_data_list) and (file_name not in bad_data_list):
        im_temp = cv2.imread(file_name)
        im_bs, im_wp = calibration(im_temp, ref)
        image_2_save = np.array(im_bs)
        contact = contact_detection(im_wp.astype(np.float32), ref_warp.astype(np.float32),20,50)
#        plt.imshow((contact*0.05+im_bs*0.95).astype(np.uint8))
        mask = contact[:, :, 1].astype(np.uint8)

        if np.sum(mask)/255 > 625.:
            # im2, contours, hierarchy = cv2.findContours(mask, 1, 2)
            contours, hierarchy = cv2.findContours(mask, 1, 2)
            (x, y), radius = cv2.minEnclosingCircle(contours[0])
            center = (int(x), int(y))
            radius = int(radius)

            if radius > 20:
                cv2.circle(im_bs, center, radius, (0, 255, 0), 2)
                cv2.imshow('image', im_bs)

                # We check if it was well labelled
                if is_circle_inside(center, radius, im_bs):
                    if int(cv2.waitKey(0)) == ok_key:
                        print iteration
                        good_data_list.append(file_name)  # We add the path to the good data image_list

                        image_shape = image_2_save.shape
                        # We compute the gradients
                        labeller = Labeller()
                        x_grad, y_grad = labeller.get_gradient_matrices(
                            center=center,
                            radius=radius,
                            size=(image_shape[0], image_shape[1]))

                         # We save everything to its folder
                        name = str(iteration).zfill(5)
                        cv2.imwrite(input_path + name + ".png", image_2_save)
                        np.save(label_path + 'x_' + name + ".npy", x_grad)
                        np.save(label_path + 'y_' + name + ".npy", y_grad)
                        iteration += 1

                        # Just in case we save the good_data_list:
                        np.save("data/processed2/good_data_list.npy", good_data_list)
                    else:
                        bad_data_list.append(file_name)
                        np.save("data/processed2/bad_data_list.npy", bad_data_list)
                        print "Discarted"
                else:
                    bad_data_list.append(file_name)
                    np.save("data/processed2/bad_data_list.npy", bad_data_list)
                    print "Circle out of limits"
        # else:
        #     print "Already sorted: " + file_name

#     good_in_good = 0
#     good_in_bad = 0
#     bad_in_good = 0
#     bad_in_bad = 0
#     visited_good = 0
#     visited_bad = 0
#     good_n = len(good_data_list)
#     bad_n = len(bad_data_list)
#     for file_path in files:
#         file_name = root+'/'+file_path
#         im_temp = cv2.imread(file_name)
#         im_bs, im_wp = calibration(im_temp, ref)
#         image_2_save = np.array(im_bs)  # NOTE: I'm not sure this is the image we want to save
#         contact = contact_detection(im_wp.astype(np.float32), ref_warp.astype(np.float32),20,50)
# #        plt.imshow((contact*0.05+im_bs*0.95).astype(np.uint8))
#         mask = contact[:, :, 1].astype(np.uint8)
#
#         if np.sum(mask)/255 > 625.:
#             # im2, contours, hierarchy = cv2.findContours(mask, 1, 2)
#             contours, hierarchy = cv2.findContours(mask, 1, 2)
#             (x, y), radius = cv2.minEnclosingCircle(contours[0])
#             center = (int(x), int(y))
#             radius = int(radius)
#
#             if radius > 20:
#                 is_good = validate_result(
#                     center=center,
#                     radius=radius,
#                     image=image_2_save)
#
#                 if (file_name in good_data_list):
#                     visited_good += 1
#                     if is_good:
#                         good_in_good += 1
#                     else:
#                         bad_in_good += 1
#                 if (file_name in bad_data_list):
#                     visited_bad += 1
#                     if is_good:
#                         good_in_bad += 1
#                     else:
#                         bad_in_bad += 1
#
#                 print "1 Alg->GOOD, Human->GOOD: " + str(float(good_in_good)/float(visited_good+_EPS))
#                 print "0 Alg->GOOD, Human->BAD: " + str(float(good_in_bad)/float(visited_bad+_EPS))
#                 print "1 Alg->BAD, Human->BAD: " + str(float(bad_in_bad)/float(visited_bad+_EPS))
#                 print "0 Alg->BAD, Human->GOOD: " + str(float(bad_in_good)/float(visited_good+_EPS))
#                 print visited_bad
#                 print visited_good
#                 print "****"
                # iteration += 1

    # bad_lights_list = average_light
    # average_light = []
    # print visited_good
    # print visited_bad
    # print good_n
    # print bad_n
    # print np.mean(bad_lights_list)
    # plt.boxplot(bad_lights_list)
    # plt.show()
