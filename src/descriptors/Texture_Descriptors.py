from typing import Dict, List
from core.CoreImage import CoreImage, Paint
from descriptors.Descriptors import FeatureExtractors

from preprocessing.Color_Preprocessor import *


from typing import *

import numpy as np
import cv2


def zigzag(input): #https://github.com/getsanjeev/compression-DCT/blob/master/zigzag.py
    #initializing the variables
    #----------------------------------
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    vmax = input.shape[0]
    hmax = input.shape[1]
    
    #print(vmax ,hmax )

    i = 0

    output = np.zeros(( vmax * hmax))
    #----------------------------------

    while ((v < vmax) and (h < hmax)):
    	
        if ((h + v) % 2) == 0:                 # going up
            
            if (v == vmin):
            	#print(1)
                output[i] = input[v, h]        # if we got to the first line

                if (h == hmax - 1):
                    v = v + 1
                else:
                    h = h + 1                        

                i = i + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                #print(2)
                output[i] = input[v, h] 
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                #print(3)
                output[i] = input[v, h] 
                v = v - 1
                h = h + 1
                i = i + 1

        
        else:                                    # going down

            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                #print(4)
                output[i] = input[v, h] 
                h = h + 1
                i = i + 1
        
            elif (h == hmin):                  # if we got to the first column
                #print(5)
                output[i] = input[v, h] 

                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1

                i = i + 1

            elif ((v < vmax -1) and (h > hmin)):     # all other cases
                #print(6)
                output[i] = input[v, h] 
                v = v + 1
                h = h - 1
                i = i + 1




        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            #print(7)        	
            output[i] = input[v, h] 
            break

    #print ('v:',v,', h:',h,', i:',i)
    return output


def kernel_generator():
    filters = []

    num_filters = 16
    ksize = 35  # The local area to evaluate
    sigma = 3 # Larger Values produce more edges
    lambd = 10
    gamma = 0.5
    psi = 0  # Offset value - lower generates cleaner results

    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization  
        filters.append(kern)
    
        # plt.imshow(kern, 
        #         cmap='gray') 
        # plt.title("Kernel with orientation = " + str(theta)) 
        # plt.show()

    return filters


class PyramidalGaborDescriptor(FeatureExtractors):
    def __init__(self, kernels: Callable | List = kernel_generator, canny_range: Tuple[float, float] = (100, 250), layers:int=4, channels:list = [0,1,2]):
        if callable(kernels):
            self._kernels = kernels()
        else:
            self._kernels = kernels
        self._channels = channels
        self._layers = layers
        self._canny_range = canny_range

    def __tile_descriptor(self, img:np.ndarray, tiles:int):

        h, w, channels = img.shape
        feature = np.array([])
        k_size_i = img.shape[0] // tiles
        k_size_j = img.shape[1] // tiles
        for chan in self._channels:
            for i in range(0, h - (h % tiles), k_size_i):
                for j in range(0, w - (w % tiles), k_size_j):
                    hist, _ = np.histogram(img[i:i + k_size_i, j:j + k_size_j, chan], bins=2, range=(0, 255))
                    feature = np.concatenate((feature, (hist / np.sum(hist))), axis=-1)

        return feature

    def extract(self, img:np.ndarray, colorspace:Callable, **kwargs):
        feature = np.array([])
        img = colorspace(img)

        newimage = np.zeros_like(img)
        for kern in self._kernels:
            image_filter = cv2.filter2D(img, -1, kern)
        np.maximum(newimage, image_filter, newimage)

        norm_fimg = cv2.normalize(newimage, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        formated_image = (norm_fimg * 255).astype("uint8")
        image_edge_g = cv2.Canny(formated_image, self._canny_range[0], self._canny_range[1])

        for s in range(1, self._layers+1):
            tiles = 2 ** s
            feature_s = self.__tile_descriptor(img=image_edge_g, tiles=tiles)
            feature = np.concatenate((feature, feature_s), axis=-1)

        return feature

class DCTDescriptor(FeatureExtractors):
    def __init__(self, tiles = 2, coefs = 50, channels = [0,1,2]) -> None:
        self._channels = channels
        self._tiles = tiles
        self._coefs = coefs
        print(f'Created DCTDescriptor. \n {self._channels=}, {self._tiles=}, {self._coefs=}.')

    
    def __tile_descriptor(self, img: np.ndarray):
        img = np.float32(img)

        h, w, channels = img.shape
        tiles = self._tiles
        feature = np.array([])
        k_size_i = img.shape[0] // tiles
        k_size_j = img.shape[1] // tiles
        for chan in self._channels:
            for i in range(0, h - (h % tiles), k_size_i):
                for j in range(0, w - (w % tiles), k_size_j):
                    trans = cv2.dct(img[i:i + k_size_i, j:j + k_size_j, chan])
                    desc = zigzag(trans)[:self._coefs]
                    feature = np.concatenate((feature, desc))

        return feature


    def extract(self, img: CoreImage, colorspace, **kwargs) -> Dict[str, np.ndarray]:
        feature = np.array([])

        feature = self.__tile_descriptor(img)

        return feature