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
from descriptors.Local_Descriptors import *

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
            if (len(a) == 1 and a[0] == ""):
                harmo_authors = "Unknown"
            else:
                a = a[0]
                author = (a[1:].split(",")[0]).split(" ")
                harmo_authors = utils.harmonize_tokens(author)

        if dic_authors.get(harmo_authors, None) is None:
            dic_authors[harmo_authors] = [idx]
        else:
            dic_authors[harmo_authors].append(idx)



    return BBDD_DB, dic_authors



def Process_Background_Removal(cfg: DictConfig, image: CoreImage) -> None:

    paint_extractor = get_class(cfg.preprocessing.background.method._target_)
    kwargs = cfg.preprocessing.background.method.kwargs
    paint_extractor.extract(image, **kwargs)


def Process_OCR_Extraction(cfg: DictConfig, coreimage: CoreImage):

    token_extractor = get_class(cfg.preprocessing.ocr.method._target_)

    local_authors = []


    for paint in coreimage._paintings:
        text_tokens, text_bbox =  token_extractor.extract(paint._paint)

        local_authors += text_tokens
        paint._text = text_tokens
        paint._text_bbox = text_bbox

    if len(coreimage) == 1:
        voting_text_tokens, voting_text_bbox = token_extractor.extract(coreimage._image)
        coreimage[0]._text += voting_text_tokens
        coreimage[0]._text_bbox += voting_text_bbox

        local_authors += voting_text_tokens

    return local_authors






def Process_QS_Descriptors(cfg: DictConfig, coreimage: [CoreImage]) -> None:
    Descriptor_Extractor = instantiate(cfg.descriptors.method)
    kwargs = cfg.descriptors.kwargs
    colorspace = get_object(cfg.descriptors.colorspace._target_)

    for paint in coreimage._paintings:
        image = paint._paint
        descriptor = Descriptor_Extractor.extract(image, colorspace=colorspace, **kwargs)
        paint._descriptors["descriptor"] = descriptor


