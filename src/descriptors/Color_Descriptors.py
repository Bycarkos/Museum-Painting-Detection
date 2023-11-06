from core.CoreImage import CoreImage, Paint
from descriptors.Descriptors import FeatureExtractors

from preprocessing.Color_Preprocessor import *


from typing import *

import numpy as np



import cv2

class MultiResolutionColorDescriptor(FeatureExtractors):

    def __init__(self, tiles: int=10, channels:list = [0,1,2]):
        self._tiles = tiles
        self._channels = channels

    def extract(self, image: np.ndarray, colorspace:Callable, **kwargs) -> np.ndarray:

        img = colorspace(image)
        h, w, channels = img.shape
        feature = np.array([])
        k_size_i = img.shape[0] // self._tiles
        k_size_j = img.shape[1] // self._tiles
        for chan in self._channels:
            for i in range(0, h - (h % self._tiles), k_size_i):
                for j in range(0, w - (w % self._tiles), k_size_j):
                    hist, _ = np.histogram(img[i:i + k_size_i, j:j + k_size_j, chan], **kwargs)
                    feature = np.concatenate((feature, (hist / np.sum(hist))), axis=-1)

        return feature


class PyramidalColorDescriptor(FeatureExtractors):
    def __init__(self, layers:int=4, channels:list = [0,1,2]):
        self._channels = channels
        self._layers = layers


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
        img = colorspace(img)
        for s in range(1, self._layers+1):
            tiles = 2 ** s
            feature_s = self.__tile_descriptor(img=img, tiles=tiles,  **kwargs)
            feature = np.concatenate((feature, feature_s), axis=-1)

        return feature


class Pyramidal_3D_ColorDescriptor(PyramidalColorDescriptor):

    def __init__(self, layers:int=4, sample: int = 12):
        self._bins = (sample,sample,sample)
        self._layers = layers

    def __tile_descriptor(self, img:np.ndarray, tiles:int, **kwargs):

        h, w, channels = img.shape
        feature = np.array([])
        k_size_i = img.shape[0] // tiles
        k_size_j = img.shape[1] // tiles
        for i in range(0, h - (h % tiles), k_size_i):
            for j in range(0, w - (w % tiles), k_size_j):
                image = img[i:i + k_size_i, j:j + k_size_j, :].reshape(-1, 3)
                hist, _ = np.histogramdd(image, bins=self._bins, **kwargs)
                hist = hist.flatten()
                feature = np.concatenate((feature, (hist / np.sum(hist))), axis=-1)

        return feature

    def extract(self, img:np.ndarray, colorspace:Callable, **kwargs):

        feature = np.array([])
        img = colorspace(img)
        for s in range(1, self._layers+1):
            tiles = 2 ** s
            feature_s = self.__tile_descriptor(img=img, tiles=tiles,  **kwargs)
            feature = np.concatenate((feature, feature_s), axis=-1)

        return feature
