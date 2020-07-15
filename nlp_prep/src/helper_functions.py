import json
from pycorenlp import StanfordCoreNLP
from util_functions import create_logger, timer, readWriteJson
from memory_profiler import profile
logger = create_logger()

URL = "http://localhost:6201"
NLP = StanfordCoreNLP(URL)


@timer
# @profile
def pos_tagger(text_input):

    '''This function takes text as input and results the corresponding POS tags as output

    Args:
        text_input: it takes string
    Return:
        it returns the POS tags of the input_text
    '''
    result1 = NLP.annotate(text_input, properties={'annotators': 'pos', 'outputFormat': 'json', 'timeout': 50000})
    data= readWriteJson(result1, "pos")
    pos_tags_list = []
    for ref in data['sentences']:
        for pos_tags in ref['tokens']:
            pos_tagsdict = {}
            pos_tagsdict['index'] = pos_tags['index']
            pos_tagsdict['word'] = pos_tags['word']
            pos_tagsdict['originalText'] = pos_tags['originalText']
            pos_tagsdict['pos'] = pos_tags['pos']
            pos_tags_list.append(pos_tagsdict)

    return pos_tags_list


@timer
# @profile
def ner_tagger(text_input):

    '''This function takes text as input and results the corresponding NER tags as output

    Args:
        text_input: it takes string
    Return:
        it returns the ner tags of the input_text
    '''
    result1 = NLP.annotate(text_input, properties={ 'annotators': 'ner', 'outputFormat': 'json', 'timeout': 50000})
    data= readWriteJson(result1, "ner")
    ner_tags_list = []    
    for ref in data['sentences']:
        if len(ref['entitymentions']) == 0:
            ner_value= []
            return ner_value
        else:
            for entity in ref['entitymentions']:
                ner_tagger_dict= {}
                ner_tagger_dict['tokenBegin']= entity['tokenBegin']
                ner_tagger_dict['tokenEnd']= entity['tokenEnd']
                ner_tagger_dict['text']= entity['text']
                ner_tagger_dict['ner']= entity['ner']
                ner_tags_list.append(ner_tagger_dict)
                final_ner_list= list({v['ner']:v for v in ner_tags_list}.values())

        return final_ner_list


@timer
# @profile
def coreferece_resolution(text_input):

    '''This function takes text as input and results the corresponding coreference_resolution as output

    Args:
        text_input: it takes string
    Return:
        it returns the coreference resolved text from the input_text
    '''
    result1 = NLP.annotate(text_input, properties={ 'annotators': 'coref', 'outputFormat': 'json', 'timeout': 50000})
    # global replacement
    data= readWriteJson(result1, "coref")
    dict_replacement = find_coreference(data["corefs"])
    if not dict_replacement:
        logger.info("no coreference to resolve")
        return text_input

    return resolve_coreference(data["sentences"], dict_replacement)


@timer
def find_coreference(data):
    '''This function takes coreference part of data payload from coreNLP in dict form and creates the coreference resolution dictionary

    Args:
        dictionary of coreference from StanfordCoreNLP api
    Return:
        dictionary of sentence wise coreference resolution information
    '''
    dict_replacement = {}
    
    for ref in data:
        logger.info("reference info %s" , ref)
        replacement = ''
        if (len(data) == 0):
            return 0
        
        
        for info in data[ref]:
                
                # global replacement
                if info['isRepresentativeMention'] == True:
                    replacement = info['text']
                    
                else:
                    to_be_replaced = info['text']
                    head_index = info['headIndex']
                    sentence_no = info['sentNum']
                    if sentence_no not in dict_replacement.keys():
                        dict_replacement[sentence_no] = [{'replacement':replacement, 'to_be_replaced': to_be_replaced, 'head_index': head_index}]
                    else: 
                        dict_replacement[sentence_no].append({'replacement':replacement, 'to_be_replaced': to_be_replaced, 'head_index': head_index})
    return dict_replacement

    
@timer
def resolve_coreference(data, dict_replacement):
    '''This function takes 2 arguments and returns coreference resolved sentence

    Args:
        1. dictionary of sentences from StanfordCoreNLP api
        2. dictionary of sentence wise coreference resolution information
    Return:
        coference resolved sentence 
    '''
    string = ""
    key = list(dict_replacement.keys())
    logger.info("key value %s", key)
    
    for sen_id,sentence in enumerate(data):
        
        if not dict_replacement[key[0]]:
            
            key.pop(0)
        
        if key and (sen_id+1) in key:    
            
            for tokens in sentence["tokens"]:
                if ((key[0]-1) == sen_id) and dict_replacement[key[0]] and (tokens["index"] == dict_replacement[key[0]][0]["head_index"]):
                    
                    if ',' in tokens["word"] or '.' in tokens["word"] or "'" in tokens["word"] or "?" in tokens["word"] or "!" in tokens["word"]:
                        string += dict_replacement[key[0]][0]["replacement"]
                    else:
                        if tokens["index"] == '1':
                            string += dict_replacement[key[0]][0]["replacement"]
                        else:
                            string += ' '+dict_replacement[key[0]][0]["replacement"]
                    dict_replacement[key[0]].pop(0)     # remove the replaced word info 
                else:
                    
                    if ',' in tokens["word"] or '.' in tokens["word"] or "'" in tokens["word"] or "?" in tokens["word"] or "!" in tokens["word"]:
                        string += tokens["word"]
                    else:
                        if tokens["index"] == '1':
                            string += tokens["word"]
                        else:
                            string += ' '+tokens["word"]
            
        else:
            
            for tokens in sentence["tokens"]:
                
                # print(tokens["index"], "\t", tokens["word"], "\t",)
                if ',' in tokens["word"] or '.' in tokens["word"] or "'" in tokens["word"] or "?" in tokens["word"] or "!" in tokens["word"]:   
                    string += tokens["word"] 
                else:
                    if tokens["index"] == '1':
                        string += tokens["word"]
                    else:
                        string += ' '+tokens["word"]
                        # corefer_res = {}
                        # corefer_res['coreferenceReplacer'] = string

    return string


