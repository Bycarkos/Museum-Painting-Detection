from preprocessing.Preprocessors import Preprocessors
from core.CoreImage import Image, Paint
from preprocessing.Color_Preprocessor import Color_Preprocessor
from preprocessing.Noise_Extractor_Preprocessor import *

import utils.utils as utils
import numpy as np

from skimage import filters
from typing import *


import cv2
from utils import *



class GF_Paint_Extractor(Preprocessors):

    @staticmethod
    def paint_bfs(img:np.ndarray):
        img2 = img.copy()
        h, w = img2.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)

        cv2.floodFill(img2, mask, (1, 1), 255)
        inv = cv2.bitwise_not(img2)

        return inv

    @classmethod
    def extract(cls, Im:Type[Image], **kwargs):
        image = Im.image
        image = Color_Preprocessor.convert2rgb(image) # image in RGB

        if utils.estimate_noise(image) > 1:
            image = NLMeans_Noise_Preprocessor.denoise(image)

        gabor_filters = utils.create_gaborfilter_bank(**kwargs)
        gf_image = utils.apply_gaborfilter_bank(image, gabor_filters)
        Im.add_transform("gabor_image", gf_image)

        edge = utils.Sobel_magnitude(gf_image, 1.5, 1.5).mean(axis=2)
        thresh = filters.threshold_otsu(edge)
        binary_image = utils.convert2image(edge>thresh)
        binary_image[0:5, :] = 0
        binary_image[:, 0:5] = 0
        binary_image[-5:, :] = 0
        binary_image[:, -5:] = 0

        painted = cls.paint_bfs(binary_image)

        mask = utils.convert2image((painted - binary_image) > 254)

        mask = utils.apply_closing(mask, (10, 10))

        ## Extract Contourns (the paintings)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        decission = []
        heigh_im, width_im = binary_image.shape

        for contour in contours:
            convexHull = cv2.convexHull(contour)

            perimeter = cv2.arcLength(convexHull, True)
            x, y, w, h = cv2.boundingRect(convexHull)
            aspect_ratio = w / h
            area = w * h
            proportion_height = h / heigh_im
            proportion_width = w / width_im

            if (proportion_height > 0.15) and (proportion_width > 0.15):
                decission.append(([y, x, h, w], perimeter, area, aspect_ratio))

        decission = sorted(decission, key=lambda x: x[2], reverse=True)
        decission = utils.non_maximun_supression(decission)

        if len(decission) > 3:
            decission = decission[:3]

        for idx, dec in enumerate(decission):
            y, x, h, w = dec[0]

            paint = Paint(image[y:y + h, x:x + w], mask=mask)
            paint.mask_bbox = dec[0]
            Im._paintings.append(paint)


        pass
