import numpy as np
import textdistance

import utils.utils as utils



from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from typing import *
from tqdm import tqdm


import cv2

import matplotlib.pyplot as plt


def bb_intersection_over_union(boxA, boxB):
    # Extracción de las coordenadas de los rectángulos
    y1, x1, h1, w1 = boxA
    y2, x2, h2, w2 = boxB

    # Coordenadas de las esquinas del rectángulo 1
    top_left1 = (y1, x1)
    bottom_right1 = (y1 + h1, x1 + w1)

    # Coordenadas de las esquinas del rectángulo 2
    top_left2 = (y2, x2)
    bottom_right2 = (y2 + h2, x2 + w2)

    # Cálculo de la intersección
    intersection_top_left = (max(top_left1[0], top_left2[0]), max(top_left1[1], top_left2[1]))
    intersection_bottom_right = (min(bottom_right1[0], bottom_right2[0]), min(bottom_right1[1], bottom_right2[1]))

    # Área de la intersección
    intersection_area = max(0, intersection_bottom_right[0] - intersection_top_left[0]) * \
                        max(0, intersection_bottom_right[1] - intersection_top_left[1])

    # Área de las cajas individuales
    area_box1 = h1 * w1
    area_box2 = h2 * w2

    # Unión de las áreas
    union_area = area_box1 + area_box2 - intersection_area
    #print(area_box1, area_box2, intersection_area, "First")
    #print(union_area, "Second")
    # Cálculo de IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou


def evaluate_mask(metric_dic: dict, ground_truth, queries:List[object]):
    precission = 0
    recall = 0
    f_score = 0

    for idx, image in tqdm(enumerate(queries), desc="evaluating the mask creation"):
        mask_pred = image.create_mask().flatten()
        mask_gt = cv2.imread(str(ground_truth[idx]))[:, :, 0].flatten() // 255

        precission += precision_score(mask_pred, mask_gt)
        recall += recall_score(mask_pred, mask_gt)
        f_score += f1_score(mask_pred, mask_gt)

    metric_dic["masking"]["fscore"] = f_score/len(queries)
    metric_dic["masking"]["recall"] = recall/len(queries)
    metric_dic["masking"]["precission"] = precission/len(queries)

def evaluate_object_detection(metric_dic: dict, ground_truth, queries):
    paint_detected = []
    for idx, image in tqdm(enumerate(queries), desc="evaluating the object detection"):

        mask_pred = utils.convert2image(image.create_mask())
        mask_gt = cv2.cvtColor(cv2.imread(str(ground_truth[idx])), cv2.COLOR_BGR2GRAY)

        contours, hierarchy = cv2.findContours(mask_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_pred, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        decission_gt = []
        decission_pred = []
        total_area = mask_gt.shape[0] * mask_gt.shape[1]
        for contour_gt in contours:
            convexHull = cv2.convexHull(contour_gt)
            x, y, w, h = cv2.boundingRect(convexHull)
            area = w*h

            if area/total_area > 0.05:
                decission_gt.append([y,x,h,w])

        for contour_pred  in contours_pred:
            convexHull = cv2.convexHull(contour_pred)
            x, y, w, h = cv2.boundingRect(convexHull)
            decission_pred.append([y,x,h,w])


        decission_pred = sorted(decission_pred, key=lambda x: x[1])
        decission_gt = sorted(decission_gt, key=lambda x: x[1])

        for pred, gt in zip(decission_pred, decission_gt):
            iou = bb_intersection_over_union(pred, gt)
            paint_detected.append(int(iou > 0.3))

    metric_dic["detection"]["recall"] = recall_score([1]*len(paint_detected), paint_detected)
"""


                for idx, author in enumerate(bbdd_authors):
                    similarity = textdistance.jaccard(harmo_authors, author)
                    decission.append(similarity)

                decission = sorted(decission, key=lambda x: x[1], reverse=True)[0][0]

                queries_authors.append(decission)
"""
def evaluate_ocr(metric_dic: dict, query_authors_files_gt:list, dict_bbdd_auth:list, queris:list):
    queries_authors = []
    for _,a in enumerate(query_authors_files_gt):
        #decission =[]
        with open(str(a), "r") as f:
            for line in  f.readlines():
                line = line.strip().split(",")
                if (len(line) == 1 and line[0] == ""):
                    harmo_authors = "Unknown"
                else:
                    line = line[0]
                    author = line.split(" ")
                    harmo_authors = utils.harmonize_tokens(author)

                if dict_bbdd_auth.get(harmo_authors, None) is None:
                    dict_bbdd_auth[harmo_authors] = len(dict_bbdd_auth) + 1

                queries_authors.append(dict_bbdd_auth[harmo_authors])

    bbdd_authors = dict_bbdd_auth.keys()

    results = []
    for _, image in tqdm(enumerate(queris), desc="Evaluating the ocr prediction"):
        for paint in image._paintings:
            decission = []
            for possible_auth in set(paint._text):
                for idx,author in enumerate(bbdd_authors):
                    similarity = textdistance.jaccard(possible_auth, author)
                    if similarity > 0.7:
                        decission.append((idx, similarity))
            if len(decission) != 0:
                decission = sorted(decission, key=lambda x: x[1], reverse=True)[0][0]
                results.append(decission)

            else:
                results.append(-1)
    print(queries_authors)
    print(results)
    metric_dic["ocr"]["recall"] = recall_score(queries_authors, results, average="micro")
    #metric_dic["ocr"]["cm"] = confusion_matrix(queries_authors, results)



def evaluate_retrieval(metric_dic: dict):
    pass