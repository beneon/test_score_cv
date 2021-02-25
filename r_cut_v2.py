import os
import yaml
import imutils
import functools
import matplotlib.pyplot as plt
import cv2
from region_cut import display_img_plt

img_file = os.path.join('dataset','version2.jpg')
img_file = cv2.imread(img_file)
# display_img_plt(img_file)

# get all the marking block

img_gray = cv2.cvtColor(img_file,cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray,(5,5),0)
_ret, img_thresh = cv2.threshold(img_blur,120,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cnts = cv2.findContours(img_thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts_bounding = [cv2.boundingRect(c) for c in cnts]
cnts_bounding_area = [b[2]*b[3] for b in cnts_bounding]

class Lane2CntBBox:
    def __init__(self,img,img_name='answer_sheet',left_pixel=0,top_pixel=0,left_padding=0,right_padding=0,compare_method='left-right'):
        """

        :param img: img cv2 object
        :param img_name:
        :param left_pixel: x of top-left pix of the lane
        :param top_pixel: y of top-left pix of the lane
        :param left_padding: left blank area width in pixel
        :param right_padding: right blank area width in pixel
        :param compare_method:
        """
        self.img = img
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
            # Contour(c, self.left_pixel)
        ]

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

class ImageObj:
    def __init__(self,image_path:str):
        assert os.path.exists(image_path)
        self.image_path = image_path
        self.image_full = cv2.imread(image_path)
        self.image_width = self.image_full.shape[1]
        self.image_height = self.image_full.shape[0]

