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


def refine_mask(image):
    # Enhancement of the external edges
    rg_chrom = Color_Preprocessor.convert2rg_chromaticity(image)
    enhanced = ((rg_chrom - utils.sharpening(rg_chrom)) * 255).astype("uint8")
    enhanced = (enhanced[:, :, 0] + enhanced[:, :, 1]) // 2

    ## applying the derivates (sobel)
    edge = utils.getGradientMagnitude(enhanced, x_importance=5.5, y_importance=5.5)

    thr = filters.threshold_otsu(edge)
    edge = (edge > thr).astype(np.uint8)

    ## Apply hough transform
    mask = np.zeros_like(edge)
    min_shape = min(edge.shape[0], edge.shape[1])
    max_line_gap = int(min_shape * 0.02)
    h_, w_ = edge.shape
    votes_min_l = int(min(h_ * 0.05, w_ * 0.05))

    linesP = cv2.HoughLinesP(edge, 1, np.pi / 180, votes_min_l, minLineLength=votes_min_l, maxLineGap=max_line_gap)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(mask, (l[0], l[1]), (l[2], l[3]), 255, 3, cv2.LINE_AA)

    # Getting the contour
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    decission = []
    heigh_im, width_im = edge.shape


    ## Getting the final bbox
    new_mask = np.zeros_like(edge)

    for contour in contours:
        convexHull = cv2.convexHull(contour)

        perimeter = cv2.arcLength(convexHull, True)
        x, y, w, h = cv2.boundingRect(convexHull)
        aspect_ratio = w / h
        area = w * h
        proportion_height = h / heigh_im
        proportion_width = w / width_im

        if (proportion_height > 0.15) and (proportion_width > 0.15) and width_im:
            decission.append(([y, x, h, w], perimeter, area, aspect_ratio))

    decission = sorted(decission, key=lambda x: x[2], reverse=True)
    decission = utils.non_maximun_supression(decission)

    new_bbox = decission[0][0]
    y, x, h, w = new_bbox
    new_mask[y:y + h, x:x + w] = 1

    return new_bbox, new_mask

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
        mask = utils.apply_dilate(mask, (5,5))

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

        for paint in Im._paintings:
            final_mask = np.zeros_like(paint._mask)
            new_bbox, new_mask = refine_mask(paint._paint)
            old_bbox = paint._mask_bbox

            yn, xn, hn, wn = new_bbox
            new_y =  (old_bbox[0] + new_bbox[0])
            new_x =  (old_bbox[1] + new_bbox[1])
            h = new_bbox[-2]
            w = new_bbox[-1]

            final_mask[new_y: new_y + h, new_x: new_x + w]

            paint._paint = paint._paint[yn:yn+hn, xn:xn+wn]
            paint._mask = final_mask
            paint._mask_bbox = new_bbox
