import numpy as np

from preprocessing.Preprocessors import Preprocessors

import cv2

class NLMeans_Noise_Preprocessor(Preprocessors):
    @classmethod
    def denoise(cls, image:np.ndarray):
        (r, g, b) = cv2.split(image)

        b_blur = cv2.medianBlur(b, 3)
        g_blur = cv2.medianBlur(g, 3)
        r_blur = cv2.medianBlur(r, 3)

        b_denoise = cv2.fastNlMeansDenoising(b_blur, 5, 9, 21)
        g_denoise = cv2.fastNlMeansDenoising(g_blur, 5, 9, 21)
        r_denoise = cv2.fastNlMeansDenoising(r_blur, 5, 9, 21)

        enhanced = cv2.merge((r_denoise, g_denoise, b_denoise))

        return enhanced