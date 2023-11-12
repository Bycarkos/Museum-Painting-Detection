from __future__ import annotations

import os
import cv2

import numpy as np

from typing import *
from PIL.Image import  Image
from pathlib import Path

from abc import ABC
from dataclasses import dataclass, field


class CoreImage():

    def __init__(self, path: Path):
        self._path = path
        self._name = os.path.basename(path).split(".")[0] # get the name of the image
        self._image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        self._paintings = []
        self._image_transforms = {}
        self._mask = None


    @property
    def image(self):
        return self._image

    def __len__(self):
        return len(self._paintings)

    def get_mask_absolut_coordenates(self, paint:Type[Paint]):
        pass

    def create_mask(self):
        mask = np.zeros(self.image.shape[:2])

        for paint in self._paintings:
            bbox = paint._mask_bbox
            y, x, h, w = bbox

            mask[y:y+h, x:x+w] = 1

        self._mask = mask

        return mask

    def add_transform(self, key, img):
        self._image_transforms[key] = img


    def __getitem__(self, item):
        return  self._paintings[item]



class Paint(CoreImage):

    def __init__(self, image:np.ndarray, mask:np.ndarray):
    ## initialize data

        self._paint: np.ndarray = image
        self._text: np.ndarray = []
        self._text_bbox = []
        self._mask: np.ndarray = mask
        self._inference: Dict[List] = {"result":None, "scores":None}
        self._candidates: List = []
        self._paint_transforms: Dict =  {}
        self._descriptors: Dict = {}
        self._mask_bbox = []


    @property
    def mask_bbox(self):
        return self._mask_bbox

    @mask_bbox.setter
    def mask_bbox(self, bbox):
        self._mask_bbox = bbox


    @property
    def paint(self):
        return self._paint











