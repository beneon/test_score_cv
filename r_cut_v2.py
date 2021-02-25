import os
import yaml
import imutils
import functools
import matplotlib.pyplot as plt
import cv2
from region_cut import display_img_plt
class ImageObj:
    def __init__(self,img):
        self.img = img
        self.img_width = self.img.shape[1]
        self.img_height = self.img.shape[0]

    def display_image(self):
        plt.imshow(self.img,cmap='gray')

class ImageFull(ImageObj):
    def __init__(self,image_path:str):
        assert os.path.exists(image_path)
        self.image_path = os.path.split(image_path)[0]
        self.image_filename = os.path.split(image_path)[1]
        cwd = os.getcwd()
        os.chdir(self.image_path)
        img = cv2.imread(self.image_filename)
        os.chdir(cwd)
        super().__init__(img)



class Lane2CntBBox:
    def __init__(self, img, img_name='answer_sheet', left_pixel=0, top_pixel=0, left_padding=0, right_padding=0, compare_method='left-right'):
        """

        :param img: img cv2 object
        :param img_name:
        :param left_pixel: x of top-left pix of the lane
        :param top_pixel: y of top-left pix of the lane
        :param left_padding: left blank area width in pixel
        :param right_padding: right blank area width in pixel
        :param compare_method:
        """
        self.img = img.img
        self.img_width = img.img_width
        self.img_height = img.img_height
        self.img_name = img_name
        self.left_pixel = left_pixel
        self.top_pixel = top_pixel
        self.left_padding = left_padding
        self.right_padding = right_padding
        self.compare_method = compare_method
        self.gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        self.blurred = cv2.GaussianBlur(self.gray,(5,5),0)
        _ret, self.thresh = cv2.threshold(
            self.blurred, 120, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        self.cnts = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.cnts = imutils.grab_contours(self.cnts)
        self.cnts = [
            Contour(c, self.left_pixel, self.top_pixel, img_parent=self)
            for c in self.cnts
        ]
        self.cnts = [c for c in self.cnts if c.is_answer()]
        self.cnts.sort()

    def disp_gray(self, msg=None):
        if msg is None:
            msg = f'gray scale image of {self.img_name}'
        cv2.imshow(msg, self.gray)

    def disp_blurred(self,msg=None):
        if msg is None:
            msg = f'blurred image of {self.img_name}'
        cv2.imshow(msg, self.blurred)

    def disp_thresh(self,msg=None):
        if msg is None:
            msg = f'threshold image of {self.img_name}'
        cv2.imshow(msg, self.thresh)

    def disp_answer_rect_marked(self,msg=None):
        _grayMarked = self.gray.copy()
        for c in self.cnts:
            cv2.rectangle(_grayMarked, (c.x, c.y), (c.x + c.w, c.y + c.h), (0, 0, 0), 2)
        if msg is None:
            msg = f'answer marked image of {self.img_name}'
        cv2.imshow(msg, _grayMarked)

class Contour:
    DEBUG=False
    w_h_ratio_min = 0.7
    w_h_ratio_max = 4
    cnt_h_min = 15
    cnt_w_min = 4
    white_pixel_whole_area_ratio_min = 0.5
    def __init__(self, c, left_pixel, top_pixel, compare_method='left-right', img_parent:Lane2CntBBox=None):
        self.img_parent = img_parent
        self.c = c
        self.x, self.y, self.w, self.h = cv2.boundingRect(self.c)
        self.actual_x = self.x+left_pixel
        self.actual_y = self.y+top_pixel
        self.actual_right_x = self.actual_x + self.w
        self.actual_bottom_y = self.actual_y + self.h
        self.w_h_ratio = float(self.w) / float(self.h)
        self.compare_method = compare_method
        self.countour_white_pixel_area = self.countour_white_pixel_area_cal()
        self.bbox_area = self.w * self.h
        self.white_pixel_whole_area_ratio = self.countour_white_pixel_area / self.bbox_area

    def __repr__(self):
        return f"({self.actual_x}, {self.actual_y}; {self.w}x{self.h})"

    def __str__(self):
        return f"({self.actual_x}, {self.actual_y}; {self.w}x{self.h})"

    def countour_white_pixel_area_cal(self):
        th_region = self.img_parent.thresh[
            self.actual_y:self.actual_bottom_y, self.actual_x:self.actual_right_x
        ]
        if self.DEBUG:
            cv2.imshow(f'th cut, {self.__str__()}', th_region)
        return cv2.countNonZero(th_region)

    def __eq__(self, other):
        # is this and other cnt overlapped at least half width and height?
        if (self.actual_x < other.actual_x and self.actual_x + self.w / 2 >= other.actual_x) and \
                (self.actual_y < other.actual_y and self.actual_y + self.h / 2 >= other.actual_y):
            return True
        else:
            return False

    def __lt__(self, other):
        if self == other:
            return False
        elif self.compare_method == "left-right":
            return self.x < other.x
        elif self.compare_method == "top-down":
            return self.y < other.y
        else:
            raise Exception(f'self.compare_method is not set correctly:{self.compare_method}')

    def is_answer(self):
        if self.w_h_ratio_max >= self.w_h_ratio >= self.w_h_ratio_min and self.w >= self.cnt_w_min and self.h >= self.cnt_h_min and self.white_pixel_whole_area_ratio > self.white_pixel_whole_area_ratio_min:
            return True
        else:
            return False



img = ImageFull(os.path.join('dataset','口腔医学专业英语补考','img1.jpg'))
img.display_image()