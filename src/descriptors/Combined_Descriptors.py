import cv2
from core.CoreImage import CoreImage, Paint
from descriptors.Descriptors import FeatureExtractors
from descriptors.Color_Descriptors import PyramidalColorDescriptor,  MultiResolutionColorDescriptor

from preprocessing.Color_Preprocessor import *

from skimage.feature import hog
from skimage.measure import  block_reduce


from typing import *

import numpy as np


class HOGPyramidal_Descriptors(PyramidalColorDescriptor):

    def __init__(self, layers:int=4, channels:list = [0,1,2], orientations:int=16,
                 cells_per_block:int=1):
        super().__init__(layers, channels)
        self._cells_block = tuple((cells_per_block, cells_per_block))
        self._orientations = orientations

    def __tile_descriptor(self, img:np.ndarray, tiles:int, **kwargs):

        h, w, channels = img.shape
        feature = np.array([])
        k_size_i = img.shape[0] // tiles
        k_size_j = img.shape[1] // tiles
        for chan in self._channels:
            for i in range(0, h - (h % tiles), k_size_i):
                for j in range(0, w - (w % tiles), k_size_j):
                    hist, _ = np.histogram(img[i:i + k_size_i, j:j + k_size_j, chan], **kwargs)
                    feature = np.concatenate((feature, (hist / np.sum(hist))), axis=-1)

        return feature

    def extract(self, img:np.ndarray, colorspace:Callable, **kwargs):

        feature = np.array([])
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        pixels_cell = int(img_gray.shape[0]*0.1), int(img_gray.shape[1]*0.1)
        hog_descriptors = hog(img_gray, orientations=self._orientations, pixels_per_cell=pixels_cell,
                	cells_per_block=self._cells_block,feature_vector=True)

        #hog_descriptors = block_reduce(hog_descriptors, block_size=max(10, int(img_gray.shape[1]*0.01)),  func=np.max)

        img = colorspace(img)
        for s in range(1, self._layers+1):
            tiles = 2 ** s
            feature_s = self.__tile_descriptor(img=img, tiles=tiles,  **kwargs)
            feature_s = block_reduce(feature_s, block_size=16, func=np.max)
            feature = np.concatenate((feature, feature_s), axis=-1)

        final_feature = (feature.reshape(-1, 1) * hog_descriptors.reshape(1, -1))
        final_feature = final_feature.mean(axis=1)

        return final_feature