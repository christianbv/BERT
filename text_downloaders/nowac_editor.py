

BASE_PATH = "/Users/christianbv/Desktop/nowac-1.1/nowac-1.1"
"""
    The Nowac file is a 3.5 GB tar file consisting of crawled norwegian webpages.
    This class aims to preprocess each file and create multiple .txt processed files in which the Bert model can
    be trained upon.

"""
class Nowac():
    def __init__(self, base_path):
        pass





def test_read(PATH):
    with open(PATH) as f:
        i = 0
        for line in f:
            if i > 1000:
                break

            print(line)
            i+=1

test_read(BASE_PATH)

