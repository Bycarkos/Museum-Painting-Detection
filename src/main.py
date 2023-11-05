# Preprocessors
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


## Auxiliar imports
from pathlib import Path
from tqdm import tqdm

import  matplotlib.pyplot as plt

import hydra
from hydra.utils import instantiate, get_class, get_object
from omegaconf import DictConfig


# Press the green button in the gutter to run the script.
@hydra.main(config_path="./configs", config_name="run", version_base="1.2")
def main(cfg:DictConfig):
    print(cfg)

    ## First part Background Removal

    if cfg.data.QS.importation.preprocessing.import_ is True:
        filepath = cfg.data.QS.importation.preprocessing.path
        QUERY_DB = utils.read_pickle(filepath)

    ## START THE PROCESS
    else:
        QUERY_IMAGES_PATHS = sorted(utils.read_bbdd(Path(cfg.data.QS.path)))
        QUERY_DB = [CoreImage(image) for image in QUERY_IMAGES_PATHS]

        if cfg.preprocessing.apply is True:
            paint_extractor = get_class(cfg.preprocessing.method._target_)
            kwargs = cfg.preprocessing.method.kwargs

            for idx, image in tqdm(enumerate(QUERY_DB), desc="Background Removal"):
                paint_extractor.extract(image, **kwargs)

            if cfg.data.QS.export.preprocessing.export_ is True:
                filepath = cfg.data.QS.export.preprocessing.path
                utils.write_pickle(information=QUERY_DB, filepath=filepath)


    ## Second Part text extraction and text detection



    ## Third parth Descriptors' extraction



    ## forth parth Retrieval and evaluation


if __name__ == "__main__":
    main()

