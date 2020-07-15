from helper_functions import coreferece_resolution,  ner_tagger , pos_tagger
from util_functions import exception
import json


@exception
def final_test(text):
    final_output = {}
    final_output['pos']= pos_tagger(text)
    final_output['ner']= ner_tagger(text)
    final_output['coref']= coreferece_resolution(text)
    yield final_output


