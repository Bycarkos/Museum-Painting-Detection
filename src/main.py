# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from core import Image as custom
from preprocessing.Paint_Extractor_Preprocessor import *


from pathlib import Path
from utils import utils

import  matplotlib.pyplot as plt

from tqdm import tqdm

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    BBDD = sorted(utils.read_bbdd(Path("data/qsd2_w1")))

    BD  = [custom.CoreImage(image) for image in BBDD]

    for idx, image in tqdm(enumerate(BD)):
        GF_Paint_Extractor.extract(image)

    utils.write_pickle(BD, "./data/qsd2_w1/qsd2_w1_processed.pkl")


    plt.imshow(BD[3][0]._paint)
    plt.show()
    #print(BD)

