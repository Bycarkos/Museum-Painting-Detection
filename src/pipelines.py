# Preprocessors
import copy
import os

from preprocessing.Preprocessors import *
from preprocessing.Paint_Extractor_Preprocessor import *
from preprocessing.Noise_Extractor_Preprocessor import *
from preprocessing.Color_Preprocessor import *
from preprocessing.Text_Extractor_Preprocessor import *

# Descriptors
from descriptors.Color_Descriptors import *
from descriptors.Text_Descriptors import *
from descriptors.Texture_Descriptors import *
from descriptors.Filtering_Descriptors import *

#CORE
from core.CoreImage import *

#Utils
from utils import utils
from utils.distance_metrics import *


## Auxiliar imports
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import  matplotlib.pyplot as plt

import hydra
from hydra.utils import instantiate, get_class, get_object
from omegaconf import DictConfig

from sklearn.metrics import precision_score, recall_score, f1_score

def Process_BBDD(cfg: DictConfig):

    if cfg.data.BBDD.importation.descriptors.import_ is False:
        BBDD_IMAGES_PATHS = sorted(utils.read_bbdd(Path(cfg.data.BBDD.path)))
        BBDD_DB = [CoreImage(image) for image in BBDD_IMAGES_PATHS]

        if cfg.descriptors.apply is True:
            Descriptor_Extractor = instantiate(cfg.descriptors.method)
            kwargs = cfg.descriptors.kwargs
            colorspace = get_object(cfg.descriptors.colorspace._target_)

            for idx, image in tqdm(enumerate(BBDD_DB), desc="Extracting descriptors from Database"):
                paint = image.image #Color_Preprocessor.convert2rgb(image.image)
                paint_object = Paint(paint, mask=np.ones_like(paint))
                image._paintings.append(paint_object)

                descriptor = Descriptor_Extractor.extract(paint, colorspace=colorspace, **kwargs)
                paint_object._descriptors["descriptor"] = descriptor

        if cfg.data.BBDD.export.descriptors.save is True:
            utils.write_pickle(BBDD_DB, filepath=cfg.data.BBDD.export.descriptors.path)
    else:
        BBDD_DB = utils.read_pickle(cfg.data.BBDD.importation.descriptors.path)


    # Group by Authors:
    BBDD_AUTHORS = sorted(utils.read_author_bbdd(Path(cfg.data.BBDD.path)))
    dic_authors = {}
    for idx, file in tqdm(enumerate(BBDD_AUTHORS), desc="Creating the Authors' Lookup Table"):
        with open(str(file), "r") as f:
            a = (f.readline().strip().split(","))
            if len(a) != 0:
                a = a[0]
                author = (a[1:].split(",")[0]).split(" ")
                harmo_authors = utils.harmonize_tokens(author)

            else:
                harmo_authors = "Unkown"

        dic_authors.get(harmo_authors, []).append(idx)


    return BBDD_DB, dic_authors



def Process_Background_Removal(cfg: DictConfig, QUERY_DB):

    paint_extractor = get_class(cfg.preprocessing.background.method._target_)
    kwargs = cfg.preprocessing.background.method.kwargs

    for idx, image in tqdm(enumerate(QUERY_DB), desc="Background Removal"):
        paint_extractor.extract(image, **kwargs)

    if cfg.preprocessing.background.export_ is True:
        filepath = os.path.join(cfg.data.QS.path, cfg.data.QN+"_processed.pkl")
        utils.write_pickle(information=QUERY_DB, filepath=filepath)

        masks_folder = os.path.join(cfg.evaluation.path, "masks")
        os.makedirs(masks_folder, exist_ok=True)

        for image in tqdm(QUERY_DB, desc="Saving the masks of the paintings"):
            new_name = image._name.split(".")[0] + ".png"
            mask = ((image.create_mask()) * 255).astype("uint8")
            filepath = os.path.join(masks_folder, new_name)
            Image.fromarray(mask).save(filepath)


def Process_OCR_Extraction(cfg: DictConfig, QUERY_DB: List[CoreImage]):
    token_extractor = get_class(cfg.preprocessing.ocr.method._target_)


    authors = []

    for idx, image in tqdm(enumerate(QUERY_DB), desc="Extracting Text With the Authors from QS"):
        local_authors = []
        for paint in image._paintings:
            token_extractor.extract(paint)
            local_authors.append(paint._text)
        authors.append(local_authors)

    print(authors)


    if cfg.preprocessing.ocr.export_ is True:
        ocr_folder = os.path.join(cfg.evaluation.path, "ocr", "authors.txt")
        os.makedirs(ocr_folder, exist_ok=True)
        utils.write_pickle(authors, filepath=ocr_folder)

        filepath = os.path.join(cfg.data.QS.path, cfg.data.QN+"_processed.pkl")
        utils.write_pickle(information=QUERY_DB, filepath=filepath)




def Process_QS_Descriptors(cfg: DictConfig, QUERY_DB: List[CoreImage]):
    Descriptor_Extractor = instantiate(cfg.descriptors.method)
    kwargs = cfg.descriptors.kwargs
    colorspace = get_object(cfg.descriptors.colorspace._target_)

    for idx, image in tqdm(enumerate(QUERY_DB), desc="Extracting descriptors from QS"):
        for paint in image._paintings:
            image = paint._paint
            descriptor = Descriptor_Extractor.extract(image, colorspace=colorspace, **kwargs)
            paint._descriptors["descriptor"] = descriptor


