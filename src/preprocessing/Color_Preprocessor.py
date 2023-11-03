from preprocessing.Preprocessors import  *



import numpy as np
import cv2


class Color_Preprocessor(Preprocessors):

    @classmethod
    def convert2rgb(cls, img:np.ndarray):
        assert len(img.shape) == 3, "This is not an 3 channel image"
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @classmethod
    def convert2lab(cls, img: np.ndarray):
        return (cv2.cvtColor(img, cv2.COLOR_BGR2LAB))

    @classmethod
    def convert2luv(cls, img: np.ndarray):
        return (cv2.cvtColor(img, cv2.COLOR_BGR2LUV))

    @classmethod
    def convert2yuv(cls, img: np.ndarray):
        return (cv2.cvtColor(img, cv2.COLOR_BGR2YUV))

    @classmethod
    def convert2hsv(cls, img: np.ndarray):
        return (cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

    @classmethod
    def convert2rg_chromaticity(cls, img: np.ndarray) -> np.ndarray:
        """
        Convert a linear RGB image to rg chromaticity color space.

        Parameters:
            img (np.ndarray): The input linear RGB image as a NumPy array.

        Returns:
            np.ndarray: The image in rg chromaticity color space as a NumPy array.
        """
        r, g, b = cv2.split(img)
        r_c = r / (img.sum(axis=2) + 1e-8)
        g_c = g / (img.sum(axis=2) + 1e-8)
        b_c = np.ones_like(r_c) - r_c
        return np.dstack((r_c, g_c, b_c))

