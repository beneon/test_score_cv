import numpy as np
import argparse
import yaml
import imutils
import functools
import matplotlib.pyplot as plt
import cv2


def display_img_plt(img):
    plt.imshow(img, cmap='gray')


class Image2Lane:
    def __init__(self, img, img_name:str='answer_sheet', thresh=None, left_pixel=0, top_pixel=0, left_padding=0, right_padding=0, compare_method='left-right', img_shape=(1,1), region_type='normal'):
        """
        cut image region of answering block into lanes
        :param img: cutted image region
        :param img_name: image str
        :param thresh: threshold data to override img converted data
        :param left_pixel: used to calculate actual x for image cut
        :param top_pixel: used to calculate actual y for image cut
        :param left_padding: left blank area
        :param right_padding: right blank area
        :param compare_method: left-right or top-down, as indicated, for contour compare
        :param img_shape: (width, height) of image region
        :param region_type: instead of inheritance and override, here we define several type and act accordingly
        """
        pass