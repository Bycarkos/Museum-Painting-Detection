import pickle
import cv2

# Preprocessors
import copy
import os

#from preprocessing.Preprocessors import *
#from preprocessing.Paint_Extractor_Preprocessor import *
#from preprocessing.Noise_Extractor_Preprocessor import *
#from preprocessing.Color_Preprocessor import *
#from preprocessing.Text_Extractor_Preprocessor import *

# Descriptors
#from descriptors.Color_Descriptors import *
#from descriptors.Text_Descriptors import *
#from descriptors.Texture_Descriptors import *
#from descriptors.Filtering_Descriptors import *

#CORE
from core.CoreImage import *

#Utils
#from utils import utils
#from utils.distance_metrics import *

#pipelines
#import pipelines as pipes

## Auxiliar imports
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import  matplotlib.pyplot as plt
from scipy.signal import convolve2d

#import hydra
#from hydra.utils import instantiate, get_class, get_object
#from omegaconf import DictConfig

#from sklearn.metrics import precision_score, recall_score, f1_score

def estimate_noise(I):

    H, W = I.shape

    M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W-2) * (H-2))

    return sigma


def CDenoissing(image):
    
    (r, g, b) = cv2.split(image)

    b_blur = cv2.medianBlur(b, 3)
    g_blur = cv2.medianBlur(g, 3)
    r_blur = cv2.medianBlur(r, 3)

    b_denoise = cv2.fastNlMeansDenoising(b_blur, 5, 9, 21)
    g_denoise = cv2.fastNlMeansDenoising(g_blur, 5, 9, 21)
    r_denoise = cv2.fastNlMeansDenoising(r_blur, 5, 9, 21)

    enhanced = cv2.merge((r_denoise, g_denoise, b_denoise))
    
    
    return enhanced

def homography(kp1, kp2, matches):
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    
    count=0
    for element in matchesMask:
        if element==1:
            count+=1
    

    return matchesMask, count

def order_database(path):
    im_list = list()
    for file in os.listdir(path):
        if '.jpg' in file:
            im_list.append(file)
    
    im_list.sort()

    return im_list

#Order database images
bbdd_list = order_database('/mnt/gpid08/users/iker.garcia/Team1/data/BBDD')

#Initialize AKAZE and BFM
akaze = cv2.AKAZE_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#Extract kp and desc from the database images
bbdd_data = list()

for image in tqdm(bbdd_list):
    img2 = cv2.imread(f'/mnt/gpid08/users/iker.garcia/Team1/data/BBDD/{image}')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_shape=img2.shape
    keypoints_2, descriptors_2 = akaze.detectAndCompute(img2,None)
    bbdd_data.append((keypoints_2, descriptors_2, img2_shape))            

with open('/mnt/gpid08/users/iker.garcia/Team1/data/processed/qsd1_w4_processed.pkl', 'rb') as f:
  bb = pickle.load(f)

matches_length = list()

for ite, image_query in enumerate(tqdm(bb, desc="Creating and saving responses for the retrieval")):
    #print(dir(image_query))
    aux_2 = list()
    if ite>0:
        for paint in image_query._paintings:
            candidates = paint._candidates
            
            aux = list()
                
            img1 = cv2.cvtColor(paint.paint, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

            if estimate_noise(gray) > 5:
                img1 = CDenoissing(img1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

            #Le sumamos la segunda derivada para enhance
            kernel = np.array([[0, -1, 0],
                            [-1, 4, -1],
                            [0, -1, 0]])
            img_derivative = cv2.filter2D(img1, -1, kernel)
            img1+=img_derivative
            img_comp = img1.copy()

            if candidates:
                for candidate in candidates:
                    #for kp_query, desc_query in quadres_data:
                    kp_bbdd, desc_bbdd, shape = bbdd_data[candidate]
                    
                    img = cv2.resize(img_comp, np.flip(shape), interpolation = cv2.INTER_LINEAR)
                    kp_query, desc_query = akaze.detectAndCompute(img,None)

                    num_matches=0
                    try:
                        #feature matching
                        matches = bf.match(np.asarray(desc_query),np.asarray(desc_bbdd))
                        num_matches = homography(kp_query, kp_bbdd, matches)[1]
                    except:
                        matches = [0]
                    aux.append((num_matches, candidate))
            else:
                for num, (kp_bbdd, desc_bbdd, shape) in enumerate(tqdm(bbdd_data)):
                    #for kp_query, desc_query in quadres_data:
                    img = cv2.resize(img_comp, np.flip(shape), interpolation = cv2.INTER_LINEAR)
                    kp_query, desc_query = akaze.detectAndCompute(img,None)

                    num_matches=0
                    try:
                        #feature matching
                        matches = bf.match(np.asarray(desc_query),np.asarray(desc_bbdd))
                        num_matches = homography(kp_query, kp_bbdd, matches)[1]
                    except:
                        matches = [0]
                    aux.append((num_matches, num))
                
            aux_2.append(aux)
                
            #matches_length.append(aux_2)

        with open(f'results/matches_filter/{ite}.pkl', 'wb') as file: 
            # A new file will be created 
            pickle.dump(aux_2, file)