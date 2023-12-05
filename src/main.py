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
from descriptors.Combined_Descriptors import *

#CORE
from core.CoreImage import *


#Utils
from utils import utils
from utils.distance_metrics import *


#pipelines
import pipelines as pipes
import evaluate as evaluators

## Auxiliar imports
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import  matplotlib.pyplot as plt
import  textdistance

import hydra
from hydra.utils import instantiate, get_class, get_object
from omegaconf import DictConfig

from sklearn.metrics import precision_score, recall_score, f1_score





# Press the green button in the gutter to run the script.
@hydra.main(config_path="./configs", config_name="run", version_base="1.2")
def main(cfg:DictConfig):
    print(cfg)

    ## BBDD
    BBDD_DB, dic_authors = pipes.Process_BBDD(cfg)
    utils.write_pickle(dic_authors, filepath="data/BBDD/authors.pkl")

    ## QS
    ## First part Background Removal

    if cfg.data.QS.preprocessed.import_ is True:
        filepath = os.path.join(cfg.data.QS.path, cfg.data.QN+"_processed.pkl")
        QUERY_DB = utils.read_pickle(filepath)

    else:
        ## START THE PROCESS
        QUERY_IMAGES_PATHS = sorted(utils.read_bbdd(Path(cfg.data.QS.path)))
        QUERY_DB = [CoreImage(image) for image in QUERY_IMAGES_PATHS]
        authors = []

        for coreimage in tqdm(QUERY_DB, desc="Preprocessing Pipeline"):

            ## Background Removal
            if cfg.preprocessing.background.apply is True:
                print(f"STARTING BACKGROUND REMOVAL FOR IMAGE {coreimage._name}")
                _ = pipes.Process_Background_Removal(cfg=cfg, image=coreimage)

            else:
                paint = coreimage._image
                paint_object = Paint(paint, mask=np.ones_like(paint))
                coreimage._paintings.append(paint_object)

            # Second Part Text Extraction
            if cfg.preprocessing.ocr.apply is True:
                print(f"EXTRACTING AUTHOR  FROM IMAGE {coreimage._name}")

                local_author = pipes.Process_OCR_Extraction(cfg, coreimage)

                authors.append(local_author)
                print(f"Response for the image {coreimage._name} with {len(coreimage)} paints: {local_author}")
                for paint in coreimage._paintings:
                    decission = []
                    for possible_auth in set(paint._text):
                        for idx, author in enumerate(dic_authors.keys()):
                            similarity = textdistance.jaccard(possible_auth, author)
                            if similarity > 0.7:
                                decission.append((idx, similarity))

                    decission = sorted(decission, key=lambda x: x[1], reverse=True)
                    for i in decission[:len(set(local_author))]:
                        paint._candidates += list(dic_authors.items())[i[0]][1]

            ## extract keypoints

        ## SAVING THE PIPELINE
        if cfg.data.QS.preprocessed.export_ is True:
            filepath = os.path.join(cfg.data.QS.path, cfg.data.QN + "_processed.pkl")
            utils.write_pickle(information=QUERY_DB, filepath=filepath)


        if cfg.preprocessing.ocr.export_ is True:
            print("SAVING THE AUTHORS FOR EACH PAINT")
            ocr_folder = os.path.join(cfg.evaluation.path, "ocr")
            filepath = os.path.join(ocr_folder, "authors.pkl")
            os.makedirs(ocr_folder, exist_ok=True)
            utils.write_pickle(authors, filepath=filepath)

        if cfg.preprocessing.background.export_ is True:
            print("SAVING THE MASKS FROM EACH IMAGE")
            masks_folder = os.path.join(cfg.evaluation.path, "masks")
            os.makedirs(masks_folder, exist_ok=True)

            for image in tqdm(QUERY_DB, desc="Saving the masks of the paintings"):
                new_name = image._name.split(".")[0] + ".png"
                mask = ((image.create_mask()) * 255).astype("uint8")
                filepath = os.path.join(masks_folder, new_name)
                Image.fromarray(mask).save(filepath)


    if cfg.descriptors.apply  is True:
        for coreimage in tqdm(QUERY_DB, desc="descriptors Pipeline"):

            print(f"EXTRACTING DESCRIPTORS FOR IMAGE {coreimage._name}")
            pipes.Process_QS_Descriptors(cfg, coreimage)

        ## forth parth: Retrieval and evaluation at this point all the necessary to compare is in Query_DB and BBDD_DB
        ## Creating Responses

        retrieval_folder = os.path.join(cfg.evaluation.path, "retrieval")
        os.makedirs(retrieval_folder, exist_ok=True)
        distance = get_object(cfg.evaluation.retrieval.similarity)
        for image_query in tqdm(QUERY_DB, desc="Creating and saving responses for the retrieval"):
            for paint in image_query._paintings:
                results=[]
                query_descriptor = paint._descriptors["descriptor"]
                if len(paint._candidates) > 0:
                    for idx in paint._candidates:
                        compare_descriptor = BBDD_DB[idx][0]._descriptors["descriptor"]
                        result = distance(compare_descriptor, query_descriptor)
                        results.append(tuple([result, idx]))


                else:
                    for idx, (image_db) in enumerate(BBDD_DB):
                        compare_descriptor = image_db[0]._descriptors["descriptor"]
                        result = distance(compare_descriptor,query_descriptor)
                        results.append(tuple([result, idx]))

                final = sorted(list(set(results)), reverse=True)[:cfg.evaluation.retrieval.k]

                scores, idx_result = list(zip(*final))
                paint._inference["result"] = list(idx_result)
                paint._inference["scores"] = list(scores)


        ## Extracting the responses for the retrieval
        final_response = []
        for idx, img in tqdm(enumerate(QUERY_DB), desc="Generating response for the retrieval"):
            local_result = []
            for painting in img._paintings:
                local_result.append(painting._inference["result"])

            final_response += (local_result)
            utils.write_pickle(information=final_response, filepath=retrieval_folder+"/result.pkl")

    ### Evaluation
    metric = {}
    ## First Evaluate the background Removal
    if cfg.evaluation.masking.evaluate is True:
        metric["masking"] = {}
        metric["detection"] = {}
        p = Path(cfg.evaluation.masking.path)
        img_list = list(p.glob("*.png"))
        masks_to_compare = sorted(img_list)

        evaluators.evaluate_object_detection(metric_dic=metric, ground_truth=masks_to_compare, queries=QUERY_DB)
        evaluators.evaluate_mask(metric_dic=metric, ground_truth=masks_to_compare, queries=QUERY_DB)

    print(metric)
    ## Second evaluate text and ocr
    if cfg.evaluation.ocr.evaluate is True:
        metric["ocr"] = {}
        bbdd_idx_author = {a: idx for idx, a in enumerate(dic_authors.keys())}
        p = Path(cfg.evaluation.ocr.path)
        img_list = list(p.glob("*.txt"))
        query_authors_files_gt = sorted(img_list)
        evaluators.evaluate_ocr(metric_dic=metric, query_authors_files_gt=query_authors_files_gt,
                                dict_bbdd_auth=bbdd_idx_author, queris=QUERY_DB)


    ## Third evaluate the retrieval
    if cfg.evaluation.retrieval.evaluate is True:
        query_gt = utils.read_pickle(cfg.evaluation.retrieval.path)
        final_query = []
        for i in query_gt:
            for j in i:
                final_query.append([j])

        query_response = copy.copy(final_response)
        print(query_gt)
        print(query_response)
        print(len(query_response))
        print(final_query)
        result_filetered = [(response, qgt) for response, qgt in zip(query_response, final_query) if qgt[0] != -1]
        final_response, final_query = list(zip(*result_filetered))
        print(final_query)
        print(final_response)
        for k in [1,2, 3, 5, 10]:
            metric[f"mapk@{str(k)}"] = utils.mapk(final_query, final_response, k)




    print(metric)
if __name__ == "__main__":
    main()

