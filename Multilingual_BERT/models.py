import torch
from torch import nn

class Bert_MultiLabler(nn.module):
    """
           Class for multilabler BERT.
           Uses normal BertModel in bottom, a dropout layer and a linear classifier on top.

           If param loadTrained is True, the model loads all three layers.
           If param loadTrained is False, the model can either load a finetuned BERT layer with empty layers on top,
           or just a pretrained BERT layer with empty layers on top.

       """
    def __init__(self):
        super(Bert_MultiLabler,self).__init__()



    pass
