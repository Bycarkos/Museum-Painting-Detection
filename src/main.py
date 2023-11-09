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



# Press the green button in the gutter to run the script.
@hydra.main(config_path="./configs", config_name="run", version_base="1.2")
def main(cfg:DictConfig):
    print(cfg)

    ## BBDD
    if cfg.data.BBDD.importation.descriptors.import_ is False:
        BBDD_IMAGES_PATHS = sorted(utils.read_bbdd(Path(cfg.data.BBDD.path)))
        BBDD_DB = [CoreImage(image) for image in BBDD_IMAGES_PATHS]

        if cfg.descriptors.color_descriptor.apply is True:
            CDescriptor_Extractor = instantiate(cfg.descriptors.color_descriptor.method)
            kwargs = cfg.descriptors.color_descriptor.kwargs
            colorspace = get_object(cfg.descriptors.color_descriptor.colorspace._target_)

            for idx, image in tqdm(enumerate(BBDD_DB), desc="Extracting descriptors from Database"):
                paint = image.image #Color_Preprocessor.convert2rgb(image.image)
                paint_object = Paint(paint, mask=np.ones_like(paint))
                image._paintings.append(paint_object)

                descriptor = CDescriptor_Extractor.extract(paint, colorspace=colorspace, **kwargs)
                paint_object._descriptors["descriptor"] = descriptor
        if cfg.descriptors.texture_descriptor.apply is True:
            TDescriptor_Extractor = instantiate(cfg.descriptors.texture_descriptor.method)

            for idx, image in tqdm(enumerate(BBDD_DB), desc="Extracting descriptors from Database"):
                paint = image.image #Color_Preprocessor.convert2rgb(image.image)
                paint_object = Paint(paint, mask=np.ones_like(paint))
                image._paintings.append(paint_object)

                descriptor = TDescriptor_Extractor.extract(paint)
                paint_object._descriptors["descriptor"] = descriptor

        if cfg.data.BBDD.export.descriptors.save is True:
            utils.write_pickle(BBDD_DB, filepath=cfg.data.BBDD.export.descriptors.path)


    else:
        BBDD_DB = utils.read_pickle(cfg.data.BBDD.importation.descriptors.path)


    ## QS

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
        else:
            for idx, image in tqdm(enumerate(QUERY_DB), desc = "Getting the images"):
                paint = image._image
                paint_object = Paint(paint, mask=np.ones_like(paint))
                image._paintings.append(paint_object)


    ## Second Part text extraction and text detection




    ## Third parth Descriptors' extraction from Query DB

    ## Color Descriptor
    if cfg.descriptors.color_descriptor.apply is True:
        CDescriptor_Extractor = instantiate(cfg.descriptors.color_descriptor.method)
        kwargs = cfg.descriptors.color_descriptor.kwargs
        colorspace = get_object(cfg.descriptors.color_descriptor.colorspace._target_)

        for idx, image in tqdm(enumerate(QUERY_DB), desc="Extracting descriptors from QS"):
            for paint in image._paintings:
                image = paint._paint
                descriptor = CDescriptor_Extractor.extract(image, colorspace=colorspace, **kwargs)
                paint._descriptors["descriptor"] = descriptor
    if cfg.descriptors.texture_descriptor.apply is True:
        TDescriptor_Extractor = instantiate(cfg.descriptors.texture_descriptor.method)

        for idx, image in tqdm(enumerate(QUERY_DB), desc="Extracting descriptors from QS"):
            for paint in image._paintings:
                image = paint._paint
                descriptor = TDescriptor_Extractor.extract(image)
                paint._descriptors["descriptor"] = descriptor





    ## forth parth: Retrieval and evaluation at this point all the necessary to compare is in Query_DB and BBDD_DB
    ## Creating Responses
    ## TODO canviar això per quan hi hagi més descriptors

    retrieval_folder = os.path.join(cfg.evaluation.path, "retrieval")
    os.makedirs(retrieval_folder, exist_ok=True)
    distance = get_object(cfg.evaluation.retrieval.similarity)

    for image_query in tqdm(QUERY_DB, desc="Creating and saving responses for the retrieval"):
        for paint in image_query._paintings:
            results=[]
            query_descriptor = paint._descriptors["descriptor"]
            for idx, (image_db) in enumerate(BBDD_DB):
                compare_descriptor = image_db[0]._descriptors["descriptor"]
                result = distance(compare_descriptor,query_descriptor)
                results.append(tuple([result, idx]))

            final = sorted(results, reverse=True)[:cfg.evaluation.retrieval.k]

            scores, idx_result = list(zip(*final))
            paint._inference["result"] = list(idx_result)
            paint._inference["scores"] = list(scores)


    ## Extracting the responses for the retrieval
    final_response = []
    for idx, img in tqdm(enumerate(QUERY_DB), desc="Generating response for the retrieval"):
        local_result = []
        for painting in img._paintings:
            local_result += (painting._inference["result"])

        final_response.append(local_result)

    if cfg.data.QS.export.descriptors.export_ is True:
        utils.write_pickle(information=final_response, filepath=retrieval_folder+"/result.pkl")


    ## Extracting the masks:
    if cfg.data.QS.export.preprocessing.export_ is True:
        masks_folder = os.path.join(cfg.evaluation.path, "masks")
        os.makedirs(masks_folder, exist_ok=True)

        for image in tqdm(QUERY_DB, desc="Saving the masks of the paintings"):
            new_name = image._name.split(".")[0] + ".png"
            mask = ((image.create_mask())*255).astype("uint8")
            filepath = os.path.join(masks_folder, new_name)
            Image.fromarray(mask).save(filepath)





    ### Evaluation
    metric = {}
    ## First Evaluate the background Removal
    if cfg.evaluation.masking.evaluate is True:
        metric["masking"] = {}
        p = Path(cfg.evaluation.masking.path)
        img_list = list(p.glob("*.png"))
        masks_to_compare = sorted(img_list)
        precission = 0
        f_score = 0
        recall = 0
        for idx, image in tqdm(enumerate(QUERY_DB), desc="evaluating the mask creation"):
            mask_pred = image.create_mask().flatten()
            mask_gt = cv2.imread(str(masks_to_compare[idx]))[:,:,0].flatten()//255

            precission += precision_score(mask_pred, mask_gt)
            recall += recall_score(mask_pred, mask_gt)
            f_score += f1_score(mask_pred, mask_gt)

        metric["masking"]["fscore"] = f_score/len(QUERY_DB)
        metric["masking"]["recall"] = recall/len(QUERY_DB)
        metric["masking"]["precission"] = precission/len(QUERY_DB)

    ## Second evaluate text and ocr


    ## Third evaluate the retrieval
    if cfg.evaluation.retrieval.evaluate is True:
        query_gt = utils.read_pickle(cfg.evaluation.retrieval.path)
        query_response = copy.copy(final_response)
        print(query_gt)
        print(query_response)
        for k in [1, 3, 5, 10]:
            metric[f"mapk@{str(k)}"] = utils.mapk(query_gt, query_response, k)




    print(metric)
if __name__ == "__main__":
    main()

