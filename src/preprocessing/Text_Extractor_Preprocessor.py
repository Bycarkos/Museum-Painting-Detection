from preprocessing.Preprocessors import Preprocessors
from core.CoreImage import CoreImage, Paint
from preprocessing.Color_Preprocessor import Color_Preprocessor
from preprocessing.Noise_Extractor_Preprocessor import *


import utils.utils as utils
import numpy as np

from skimage import filters
from typing import *


import cv2
import easyocr
import copy
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt


class Fourier_Token_Extractor(Preprocessors):

        @classmethod
        def extract(cls, image: np.ndarray):

            if utils.estimate_noise(image) > 1:
                image = NLMeans_Noise_Preprocessor.denoise(image)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            comb = utils.scharr_filter(gray, True)

            kernel = np.ones((3, 1))
            binary = cv2.medianBlur(cv2.morphologyEx(comb, cv2.MORPH_OPEN, kernel), 7)
            binary = (binary[:, :, None]).mean(axis=2)

            img_fourier = np.fft.fftshift(np.fft.fft2(binary))
            mag_ff = np.log(abs(img_fourier))
            binary_img = copy.copy(img_fourier)

            ## horitzontal filter pass
            center_coordinates = (mag_ff.shape[1] // 2, mag_ff.shape[0] // 2)

            axes_length = (int(mag_ff.shape[1] * 0.05), int(mag_ff.shape[0] * 0.05))

            mask_image = np.zeros_like(mag_ff).copy()

            horitzontal_elipsis = cv2.ellipse(mask_image, center_coordinates, axes_length, 0, 0, 360, (255, 255), -1)

            ## vertical filter band
            center_coordinates = (mag_ff.shape[1] // 2, mag_ff.shape[0] // 2)

            axes_length = (int(mag_ff.shape[0] * 0.01), int(mag_ff.shape[1] * 0.01))

            mask_image = np.zeros_like(mag_ff).copy()

            vertical_elipsis = cv2.ellipse(mask_image, center_coordinates, axes_length, 0, 0, 360, (255, 255), -1)

            ## apply butterfilter
            ff_h = binary_img * horitzontal_elipsis
            ff_v = binary_img * vertical_elipsis

            ## recover image in both directions
            img_back_h = abs(np.fft.ifft2(ff_h))
            img_back_v = abs(np.fft.ifft2(ff_v))

            ## intersect intensities in two directions
            merge = (img_back_v + img_back_h) / 2

            ## threshold
            thresholded = ((merge > merge.max() // 3) * 255).astype("uint8")

            ## dilate
            kernel = (int(thresholded.shape[1] * 0.025), 1)
            final_mask = utils.apply_dilate(thresholded, kernel=kernel, iterations=5)

            ## Detect the mask
            contours, hierarchy = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            decission = []
            for contour in contours:
                convexHull = cv2.convexHull(contour)

                x, y, w, h = cv2.boundingRect(convexHull)
                aspect_ratio = w / h
                area = w * h
                perimeter = cv2.arcLength(convexHull, True)

                # if w > h:
                #constrain_1 = (final_mask.shape[0] // 30 < y) and (10 * final_mask.shape[0] // 30 > y)
                #constrain_2 = (7 * final_mask.shape[0] // 10 < y) and (final_mask.shape[0] > y)
                #if constrain_1 or constrain_2:
                decission.append(([y, x, h, w], area, perimeter, aspect_ratio))

            decission = sorted(decission, key=lambda x: x[2], reverse=True)
            if len(decission) == 0:
                return ["None", "None"], [[-1,-1,-1,-1], [-1,-1,-1,-1]]

            ## merge all bbox aligned over same axis
            final_mask_2 = np.zeros_like(final_mask)
            ay, ax, ah, aw = decission[0][0]

            final_mask_2[ay:ay + ah, ax:ax + aw] = 1

            constrain_y = final_mask_2.shape[0] * 0.01
            constrain_x = final_mask_2.shape[1] * 0.2

            if len(decission) > 1:
                for idx in range(1, len(decission)):
                    new_y, new_x, new_h, new_w = decission[idx][0]
                    condition_w = min(abs(ax - new_x), abs(ax + aw - new_x))
                    if (abs(ay - new_y) < constrain_y) and (condition_w < constrain_x):
                        ay, ax, ah, aw = min(ay, new_y) - 10, min(ax, new_x) + 10, max(ah, new_h), (aw + new_w)

                        final_mask_2[ay:ay + ah, ax:ax + aw] = 1

                contours, hierarchy = cv2.findContours(final_mask_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                convexHull = cv2.convexHull(contours[0])

                x, y, w, h = cv2.boundingRect(convexHull)
                final_decission = [y, x, h, w]

            else:
                final_decission = decission[0][0]

            # text extraction
            reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            y, x, h, w = final_decission

            if y - 10 < 0:
                detected = gray[y:y + h + 5, x:x + w]
            else:
                detected = gray[y - 10:y + h + 5, x:x + w]

            tokens = sorted(reader.readtext(detected, detail=0))
            text_tokens = utils.harmonize_tokens(tokens)

            easy_tokens = reader.readtext(image)

            ## THe part of the easy OCR
            easy_text = [res[1] for res in easy_tokens]
            easy_bbox = [res[0] for res in easy_tokens]

            if len(easy_bbox) > 1:

                tmpy = []
                tmpx = []
                for bbox in easy_bbox:
                    for point in bbox:
                        tmpy.append(point[1])
                        tmpx.append(point[0])

                y = min(tmpy)
                x = min(tmpx)
                w = max(tmpx) - x
                h = max(tmpy) - y
                easy_tokens = easy_text
                easy_decission = [y,x,h,w]
                easy_text = utils.harmonize_tokens(easy_tokens)

            elif len(easy_bbox) != 0:

                y = easy_bbox[0][0][1]
                x = easy_bbox[0][0][0]
                w = easy_bbox[0][1][0] - x
                h = easy_bbox[0][2][1] - y
                easy_tokens = easy_tokens[0][1].split(" ")
                easy_decission = [y,x,h,w]
                easy_text = utils.harmonize_tokens(easy_tokens)

            else:
                easy_decission = [-1, -1, -1, -1]
                easy_text = None

            return [text_tokens, easy_text], [final_decission, easy_decission]

