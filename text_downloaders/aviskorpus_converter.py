import os
import xml.etree.ElementTree as ET


BASE_DIR = "/Users/christianbv/Desktop/Articles"
BASE_SAVE_PATH = "/Users/christianbv/PycharmProjects/DistilBert/texts"


"""
    Class to convert and keep track of news-articles used
    News are saved as Articles -> Years -> Papers -> article1, article2...
    
"""
class NewsCorpusConverter():
    def __init__(self):
        pass


def get_year_dirs() -> list:
    return [BASE_DIR+"/"+y for y in os.listdir(BASE_DIR) if "." not in y]

"""
    Returns all papers for a given year
    Filters on getting only bokmaal articles or nynorsk articles, or both
"""
def get_paper_dirs(year_dir, bm = False, nn = False) -> list:
    assert bm + nn != 0

    if bm and nn:
        return [year_dir + "/" + paper for paper in os.listdir(year_dir)]
    elif bm and not nn:
        return [year_dir + "/" + paper for paper in os.listdir(year_dir) if "nob" in paper]
    else:
        return [year_dir + "/" + paper for paper in os.listdir(year_dir) if "nno" in paper]

"""
    Returns all articles for a given paper
"""
def get_article_paths(dir) -> list:
    return [dir + "/" + article for article in os.listdir(dir) if ".xml" == article[-4:]]

"""
    As each paragraph is quite long, we split each "text" into two paragraphs belonging together
"""
def get_texts(path) -> list:
    texts = []
    body = ET.parse(path).getroot().find("body")
    paragraphs = [f.text for div in body for f in div if "text" in div.attrib.values()]

    # Splits up into two and two:
    print(paragraphs)
    print(sum([len(x) for x in paragraphs])) # 2297 ord..

    texts = []
    for i in range(0, len(paragraphs),2):
        texts.append(paragraphs[i] + paragraphs[i+1])

    return texts

def save_texts(texts, year):
    save_path = BASE_SAVE_PATH+"/processed_"+year

    f1 = open(save_path, "w")
    for text in texts:
        f1.write(text)
        f1.write("\n")
        f1.write("\n")

    f1.close()




years = get_year_dirs()
papers = get_paper_dirs(years[0], bm = True, nn = False)
articles_test = get_article_paths(papers[0])
foo = get_texts(articles_test[1])
