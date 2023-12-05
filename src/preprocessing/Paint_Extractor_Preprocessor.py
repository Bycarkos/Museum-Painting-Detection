from preprocessing.Preprocessors import Preprocessors
from core.CoreImage import CoreImage, Paint
from preprocessing.Color_Preprocessor import Color_Preprocessor
from preprocessing.Noise_Extractor_Preprocessor import *

import utils.utils as utils
import numpy as np

from skimage import filters
from typing import *

import matplotlib.pyplot as plt



import cv2



def refine_mask(image):
    # Enhancement of the external edges
    rg_chrom = Color_Preprocessor.convert2rg_chromaticity(image)
    enhanced = ((rg_chrom + utils.sharpening(rg_chrom)) * 255).astype("uint8")
    enhanced = (enhanced[:, :, 0] + enhanced[:, :, 1]) / 2

    ## applying the derivates (sobel)
    edge = utils.Sobel_magnitude(enhanced, x_importance=6, y_importance=6)

    thr = filters.threshold_otsu(edge)
    edge = (edge > thr).astype(np.uint8)
    k = (int(edge.shape[0] * 0.03), int(edge.shape[1] * 0.03))
    edge = utils.apply_closing(edge, k)
    edge = cv2.medianBlur(edge, 5)

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
    absolut_area = new_mask.shape[0] * new_mask.shape[1]

    for contour in contours:
        convexHull = cv2.convexHull(contour)

        perimeter = cv2.arcLength(convexHull, True)
        x, y, w, h = cv2.boundingRect(convexHull)
        aspect_ratio = w / h
        area = w * h
        proportion_height = h / heigh_im
        proportion_width = w / width_im

        if (proportion_height > 0.15) and (proportion_width > 0.15) and width_im  and (area> absolut_area * 0.7):
            decission.append(([y, x, h, w], perimeter, area, aspect_ratio))

    decission = sorted(decission, key=lambda x: x[2], reverse=True)
    decission = utils.non_maximun_supression(decission)

    if len(decission) != 0:
        new_bbox = decission[0][0]
        y, x, h, w = new_bbox
        new_mask[y:y + h, x:x + w] = 1

    else:
        new_bbox = [0,0,0,0]
        new_mask = np.ones_like(new_mask)


    return new_bbox, new_mask


class Canny_Paint_Extractor(Preprocessors):

    @staticmethod
    def paint_bfs(img:np.ndarray):
        img2 = img.copy()
        h, w = img2.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)

        cv2.floodFill(img2, mask, (1, 1), 255)
        inv = cv2.bitwise_not(img2)

        return inv

    @classmethod
    def extract(cls, Im: CoreImage, colorspace:Callable):

        ## Preprocess of the image
        img = Im.img

        if utils.estimate_noise(img) > 1:
            image = NLMeans_Noise_Preprocessor.denoise(img)

        chromacity = colorspace(img)
        chromacity = (chromacity * 255).astype(np.uint8)

        x = cv2.Sobel(chromacity, cv2.CV_64F, 1, 0, ksize=3, scale=1)
        y = cv2.Sobel(chromacity, cv2.CV_64F, 0, 1, ksize=3, scale=1)
        absx = cv2.convertScaleAbs(x)
        absy = cv2.convertScaleAbs(y)
        edge = cv2.addWeighted(absx, 1.5, absy, 1.5, 10)

        kernel = (int(edge.shape[0] * 0.01), int(edge.shape[1] * 0.01))
        edges = cv2.Canny(edge, 75, 100)

        binary = utils.apply_closing(edges, kernel)
        # binary = apply_open(binary, (3,3))
        binary[0:5, :] = 0
        binary[:, 0:5] = 0
        binary[-5:, :] = 0
        binary[:, -5:] = 0

        # Flood filling the background to remove external noise
        painted = cls.paint_bfs(binary)

        ## invert to remove background
        mask = painted + binary
        mask = utils.apply_closing(mask, (10, 10))
        mask = utils.apply_dilate(mask, (3, 3))
        mask = ((mask > 250) * 255).astype("uint8")

        ## finding the contourns

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        decission = []

        heigh_im, width_im =  binary.shape
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

            tmp_y, tmp_x, tmp_h, tmp_w = new_bbox
            old_bbox = paint._mask_bbox

            if not tmp_y == tmp_x == tmp_h == tmp_w:
                yn, xn, hn, wn = new_bbox
                new_y = (old_bbox[0] + new_bbox[0])
                new_x = (old_bbox[1] + new_bbox[1])
                h = new_bbox[-2]
                w = new_bbox[-1]

                final_mask[new_y: new_y + h, new_x: new_x + w]

                paint._paint = paint._paint[yn:yn+hn, xn:xn+wn]
                paint._mask = final_mask
                paint._mask_bbox = new_bbox


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
    def extract(cls, Im:CoreImage, **kwargs):
        image = Im.image

        if utils.estimate_noise(image) > 1:
            print("gola")
            image = NLMeans_Noise_Preprocessor.denoise(image)

        shaped = int(min(image.shape[0], image.shape[1]) * 0.03)
        kwargs["n_filters"] = min(kwargs["ksize"], shaped)
        gabor_filters = utils.create_gaborfilter_bank(**kwargs)
        gf_image = utils.apply_gaborfilter_bank(image, gabor_filters)
        Im.add_transform("gabor_image", gf_image)

        edge = utils.Sobel_magnitude(gf_image, 3.5, 3.5)
        edge = cv2.cvtColor(edge, cv2.COLOR_RGB2GRAY)

        thresh = filters.threshold_otsu(edge)
        binary_image = utils.convert2image(edge>thresh)
        binary_image = utils.apply_closing(binary_image, (10, 10))

        binary_image[0:5, :] = 0
        binary_image[:, 0:5] = 0
        binary_image[-5:, :] = 0
        binary_image[:, -5:] = 0

        painted = cls.paint_bfs(binary_image)

        mask = (painted - binary_image)
        mask = utils.apply_closing(mask, (15, 5))
        mask = utils.apply_open(mask, (1, 10))
        mask = utils.apply_closing(mask, (20, 20))
        mask = utils.convert2image(mask > 250)
        mask = binary_image + mask
        mask = cv2.medianBlur(mask, 5)
        mask = utils.apply_open(mask, (10, 20))

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
        decission = sorted(utils.non_maximun_supression(decission), key=lambda x: x[0][1])

        if len(decission) > 3:
            decission = decission[:3]

        if len(decission) == 0:
            decission.append(([0,0,mask.shape[1], mask.shape[0]], -1, -1, -1))

        for idx, dec in enumerate(decission):
            y, x, h, w = dec[0]

            paint = Paint(image[y:y + h, x:x + w], mask=mask)
            paint.mask_bbox = dec[0]
            Im._paintings.append(paint)

        for paint in Im._paintings:
            final_mask = np.zeros_like(paint._mask)
            new_bbox, new_mask = refine_mask(paint._paint)

            tmp_y, tmp_x, tmp_h, tmp_w = new_bbox
            old_bbox = paint._mask_bbox

            if not tmp_y == tmp_x == tmp_h == tmp_w:
                yn, xn, hn, wn = new_bbox
                new_y = (old_bbox[0] + new_bbox[0])
                new_x = (old_bbox[1] + new_bbox[1])
                h = new_bbox[-2]
                w = new_bbox[-1]

                final_mask[new_y: new_y + h, new_x: new_x + w] = 1

                paint._paint = paint._paint[yn:yn+hn, xn:xn+wn]
                paint._mask = final_mask
                paint._mask_bbox = [new_y, new_x, h, w]

