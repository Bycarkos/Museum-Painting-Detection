

import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List
from PIL.Image import Image
from PIL import Image as Im
from skimage import filters
from scipy.signal import convolve2d
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from typing import *


import os
import cv2
import copy
import pickle


def write_pickle(information:Any,filepath:str):
    """
    Serialize and write an object to the specified file using pickle.

    Parameters:
        information (Any): The object to be serialized and written.
        filepath (str): The path to the file to write the pickled object.
    """

    abs_path = os.path.dirname(filepath)
    os.makedirs(abs_path, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(information, f)
def read_bbdd(path: Type[str]) -> List[Type[Path]]:
    """
    Reads image files from the specified directory path and returns a list of image file names.

    Args:
        path (Type[str]): The directory path where the images are located.
        x (List[int, ...]): Unused argument (appears to be a placeholder).

    Returns:
        List[Path, ...]: A list of image file names (with the .jpg extension) in the specified directory.
    """

    p = Path(path)
    img_list = list(p.glob("*.jpg")) # lista [~/BBDD/bbdd_0000.jpg ...]
    return img_list

def estimate_noise(img: np.ndarray):

    if len(img.shape) > 2:
        img = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

    H, W = img.shape[:2]

    M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(img, M))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W-2) * (H-2))

    return sigma



def apply_gaborfilter_bank(img:np.ndarray, filters:List[np.ndarray]):
    # This general function is designed to apply filters to our image

    # First create a numpy array the same size as our input image
    newimage = np.zeros_like(img)

    # Starting with a blank image, we loop through the images and apply our Gabor Filter
    # On each iteration, we take the highest value (super impose), until we have the max value across all filters
    # The final image is returned
    depth = -1  # remain depth same as original image

    for kern in filters:  # Loop through the kernels in our GaborFilter
        image_filter = cv2.filter2D(img, depth, kern)  # Apply filter to image

        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        np.maximum(newimage, image_filter, newimage)
    return newimage

def Sobel_magnitude(im, x_importance:float=1.5, y_importance:float=1.5):
    "Get magnitude of gradient for given image"
    ddepth = cv2.CV_64F
    dx = cv2.Sobel(im, ddepth, 1, 0)
    dy = cv2.Sobel(im, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, x_importance, dyabs, y_importance, 0)
    return mag

def create_gaborfilter_bank(**kwargs):
    # This function is designed to produce a set of GaborFilters
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree

    filters = []
    num_filters = kwargs.get("n_filters", 20)
    ksize = kwargs.get("n_filters", 32)  # The local area to evaluate
    sigma = kwargs.get("sigma",4.0)   # Larger Values produce more edges
    lambd = 10.0
    gamma = 0.5
    psi = 0  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters

def normalize(img, m=0., mx=1.):
    return cv2.normalize(img, None, m, mx, cv2.NORM_MINMAX, dtype=cv2.CV_64F)

def convert2image(img:np.ndarray):
    return (img*255).astype("uint8")




## Morphological operations
def apply_morpholical_grad(image, kernel):
    if isinstance(kernel, tuple):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           kernel)

    op = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return op


def apply_open(image, kernel, iters: int = 3):
    if isinstance(kernel, tuple):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           kernel)

    op = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iters)
    return op


def apply_dilate(image, kernel, iterations: int = 5):
    if isinstance(kernel, tuple):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           kernel)

    op = cv2.dilate(image, kernel, iterations=iterations)
    return op


def apply_closing(image, kernel):
    if isinstance(kernel, tuple):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           kernel)

    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closing


def apply_erode(image, kernel):
    if isinstance(kernel, tuple):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           kernel)

    op = cv2.erode(image, kernel)
    return op


def check_overlap(ba, bb, threshold: float = 0.5):
    y1, x1, h1, w1 = ba
    y2, x2, h2, w2 = bb

    # Calculate the coordinates of the bounding boxes
    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    # Calculate the intersection area
    intersection_area = max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) * \
                        max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    # Calculate areas of each bounding box
    area_bbox1 = (x1_max - x1_min) * (y1_max - y1_min)
    area_bbox2 = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate areas of each bounding box
    area_bbox1 = w1 * h1
    area_bbox2 = w2 * h2

    overlap_ratio = intersection_area / min(area_bbox1, area_bbox2)
    return overlap_ratio >= threshold

def non_maximun_supression(bboxes: list, threshold: float = 0.5):
    final_voting = []
    bbox_list = copy.copy(bboxes)
    while len(bbox_list) > 0:
        current_box = bbox_list.pop(0)
        final_voting.append(current_box)
        for bbox in bbox_list:
            overlap = check_overlap(current_box[0], bbox[0])
            if overlap:
                bbox_list.remove(bbox)

    return final_voting


