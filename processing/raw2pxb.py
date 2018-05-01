import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import cv2


class RAW2PXB():
    # This class crops the background of GS images and converts them to grayscale
    def __init__(self):
        pass

    def show_image(self, img):
        fig = plt.figure()
        plt.imshow(img.astype(np.uint8), cmap='gray')
        plt.show()

    def show_2images(self, img1, img2):
        fig = plt.figure()
        a = fig.add_subplot(1,2,1)
        plt.imshow(img1.astype(np.uint8), cmap='gray')
        a.set_title('Before')

        a = fig.add_subplot(1,2,2)
        plt.imshow(img2.astype(np.uint8), cmap='gray')
        a.set_title('After')
        plt.show()

    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    def __make_kernal(self, n):
        a, b = (n-1)/2, (n-1)/2
        r = (n-1)/2
        y, x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r
        kernal = np.zeros((n, n)).astype(np.uint8)
        kernal[mask] = 255
        return kernal

    def __back_substraction(self, contact_image, noncontact_image):
        background = cv2.GaussianBlur(noncontact_image.astype(np.float32),(25,25),15).astype(np.int32)
        raw_back_sub = contact_image.astype(np.int32) - background
        img = ((raw_back_sub + round(np.mean(background)))).astype(np.uint8)
        # img = (contact_image.astype(np.int32) - noncontact_image.astype(np.int32)).astype(np.int32)
        return img

    def __contact_detection(self, im, low_bar, high_bar):
        # print low_bar, high_bar
        background = cv2.GaussianBlur(im.astype(np.float32),(25,25),15)
        im_sub = im/background*70
        # self.show_image(background, im_sub)
        im_gray = self.rgb2gray(im_sub).astype(np.uint8)
        im_canny = cv2.Canny(im_gray, low_bar, high_bar)
        # self.show_image(im_canny)
        kernal1 = self.__make_kernal(7) # How big we connect the islands into a big island (k1 <= k2)
        kernal2 = self.__make_kernal(20) #
        kernal3 = self.__make_kernal(20)
        img_d = cv2.dilate(im_canny, kernal1, iterations=1)
        img_e = cv2.erode(img_d, kernal1, iterations=1)
        img_ee = cv2.erode(img_e, kernal2, iterations=1)
        img_dd = cv2.dilate(img_ee, kernal3, iterations=1)
        img_label = np.stack((img_dd, img_dd, img_dd), axis=2).astype(np.uint8)
        return img_label

    def __contact_detection1(self, im0, im1):
        im_crop0 = im0[:-80, 40:-50]
        im_crop1 = im1[:-80, 40:-50]
        contact = self.__contact_detection(im_crop1, 20, 60)
        return im_crop0, im_crop1, contact, im_crop1.astype(np.uint8) + contact/10

    def __contact_detection2(self, im0, im1):
        length = 53.3
        width = 35
        rows, cols, cha = im1.shape
        M = np.array([[  7.73083712e-01,  -2.90975541e-01,   6.91273054e+01],
           [ -2.64321106e-03,   7.08488152e-01,   2.82502854e+00],
           [ -2.38610408e-05,  -9.15087898e-04,   1.00000000e+00]])

        warped0 = cv2.warpPerspective(im0, M, (cols, rows))
        warped1 = cv2.warpPerspective(im1, M, (cols, rows))
        im_crop0 = warped0[40:-5, 70:-50]
        im_crop1 = warped1[40:-5, 70:-50]
        contact2 = self.__contact_detection(im_crop1, 20, 60)
        return im_crop0, im_crop1, contact2, im_crop1.astype(np.uint8) + contact2/10

    def crop_contact(self, img_back, img_grasp, gel_id=1, compress_factor=1, is_zeros=True):
        if gel_id == 1:
            background, im1, contact, patch = self.__contact_detection1(img_back, img_grasp)
        else:
            background, im1, contact, patch = self.__contact_detection2(img_back, img_grasp)
        # contact = rgb2gray(contact).astype(np.uint8)
        contact = scipy.sign(contact)
        img_with_back_sub = self.__back_substraction(im1, background)
        if is_zeros:
            img_mean = np.ones(im1.shape)*0
        else:
            img_mean = np.ones(im1.shape)*np.mean(background)
        new_img = contact * img_with_back_sub + (np.ones(im1.shape) - np.ones(im1.shape)*contact) *img_mean

        no_back = cv2.cvtColor(new_img.astype('uint8'), cv2.COLOR_RGB2GRAY)

        if compress_factor != 1:
            no_back = self.resize_image(no_back, factor=float(compress_factor))
        return no_back

    def resize_image(self, img, factor):
        return cv2.resize(img, (0, 0), fx=factor, fy=factor)

    def multiple_image_processing(self, gel_id, img_back, img_list, force_list, layer_var=50, max_f_offset=100, compress_factor=0.2):
        def get_heights():
            m = (205-max_f_offset)/max(force_list)
            heights = []
            for force in force_list:
                heights.append(205-m*force)
            return heights

        def add_layer(base_layer, top_layer, height, layer_var):
            for i in range(top_layer.shape[0]):
                for j in range(top_layer.shape[1]):
                    if top_layer[i][j] != 0:
                        new_pixel = int(height + top_layer[i][j]*float(layer_var)/255.0)
                        base_layer[i][j] = max(new_pixel, base_layer[i][j])
            return base_layer

        heights = get_heights()
        no_back = self.crop_contact(img_back=img_back, img_grasp=img_list[0], gel_id=gel_id)
        base_layer = np.zeros(no_back.shape)
        for img, height in zip(img_list, heights):
            no_back = self.crop_contact(img_back=img_back, img_grasp=img, gel_id=gel_id)
            base_layer = add_layer(base_layer, no_back, height, layer_var)

        blur = cv2.GaussianBlur(base_layer, (5, 5), 50)
        blur = self.resize_image(blur, compress_factor)
        # return blur
        # self.show_image(blur)
        return cv2.flip(blur, 1)

if __name__ == "__main__":
    gs = cv2.imread('sample_data/GS1_2.png')
    gs_back = cv2.imread('sample_data/arc/gs_image2.png')
    #
    r2p = RAW2PXB()
    no_back = r2p.crop_contact(gs_back, gs, gel_id=1, compress_factor=0.2)
    r2p.show_2images(img1=gs[:-80, 40:-50], img2=no_back)
    pass
