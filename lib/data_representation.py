class InputExample(object):
    """
        object to control input
            - guid is the id of the input
            - text_a is the first sentence in the input
            - text_b is the second sentence in the input
            - labels is label(s) of the example
        - Benyttes på hver rådoc
    """

    def __init__(self, guid, text_a, text_b=None, labels=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):
    """
        Object for each tokenized doc
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


# Helper methods:
"""
    Takes in two tokens, and strips the longest token of one char until both tokens has len lower than max_len.
"""


def truncate_pairs(tok_a, tok_b, max_len):
    while True:
        tot_len = len(tok_a) + len(tok_b)
        if tot_len <= max_len:
            break
        if len(tok_a) > len(tok_b):
            tok_a.pop()
        else:
            tok_b.pop()
    return tok_a, tok_b
