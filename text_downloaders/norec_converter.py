import pandas as pd
import numpy as np
import os
import re

BASE_DIR = "/Users/christianbv/Downloads/norec_fine-master/data/"
OUT_PATH = "Users/christianbv/PycharmProjects/DistilBert/texts/norec/processed_norec.txt"

def get_filenames():
    datatypes = ["train", "test","dev"]
    res = []
    for d in datatypes:
        fs = os.listdir(BASE_DIR+d)
        for f in fs:
            if ".ann" in f:
                continue
            else:
                res.append(BASE_DIR+d+"/"+f)
    return res


def load_texts(filenames: list):
    texts = []
    for file in filenames:
        with open(file) as f:
            lines = f.readlines()
            text = []
            for i, line in enumerate(lines[1:]): # Removing title
                if str(line) == '\n' and len(text) > 1:
                    texts.append(text)
                    print(i,text)
                    text = []
                    continue

                if len(line) > 20:
                    text.append(re.sub("[^A-Za-z0-9 ,.?!()øæå]+",'', str(line)).rstrip(" ."))
                else:
                    continue


    return texts
"""
    Method to parse texts
    - removing weird chars such as '(,),\ etc.' 
    - converting to Unicode
"""
def convert_texts(texts: list) -> pd.DataFrame:
    def parse_text(text:str):
        # Removing non-ascii chars

        text = re.sub(r'[^\x00-\x7f]', r'', text)
        pass


    pass

def write_texts_to_file(filePath: str = "/Users/christianbv/PycharmProjects/DistilBert/texts/processed_norec.txt",
                        texts:list = None):
    if not texts:
        print("Did not write anything to file.")
        return

    # Writing to txt file
    f1 = open(filePath, "w")
    for text in texts:
        f1.write((". ").join(text))
        f1.write("\n")
        f1.write("\n")
    f1.close()


filenames = get_filenames()
texts = load_texts(filenames)
write_texts_to_file(texts = texts)