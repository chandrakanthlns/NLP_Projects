from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize
import json
import pandas as pd
pd.set_option('display.max_colwidth',1000)
from sklearn.metrics.pairwise import cosine_similarity
import pickle



with open('./glove.840B.50d.pkl', 'rb') as fp:
    glove = pickle.load(fp)


def read_text(text):

    """read the text from document and split the sentence"""

    sentences = sent_tokenize(text)
    return sentences



VECTOR_SIZE = 50
EMPTY_VECTOR = np.zeros(VECTOR_SIZE)
def sentence_vector(sentence):
    return sum([glove.get(word , EMPTY_VECTOR) for word in sentence])/len(sentence)

def sentences_to_vector(sentences):
    return [sentence_vector(sentence) for sentence in sentences]



def similarity_matrix(sentence_vectors):

    """calculate the similarity matrix among the sentences."""

    sim_mat = np.zeros([len(sentence_vectors), len(sentence_vectors)])
    for i in range(len(sentence_vectors)):
        for j in range(len(sentence_vectors)):
            element_i = sentence_vectors[i].reshape(1,VECTOR_SIZE)
            element_j = sentence_vectors[j].reshape(1,VECTOR_SIZE)
            sim_mat[i][j] = cosine_similarity(element_i, element_j)       
    return sim_mat


def generate_summary(text):

    """generate the summary by selecting the top sentences by graph importance"""

    summarize_text = []
    
    read_json_data = {}
    input_text = read_text(text)
    count_lines = len(input_text)
    top_n = count_lines % 4
    sentences = sentences_to_vector(input_text)
    sentence_similarity_martix = similarity_matrix(sentences)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(input_text)), reverse=True)
    for i in range(top_n):
        summarize_text.append("".join(ranked_sentence[i][1]))
        final_summarize_text = ", ".join(summarize_text)
        read_json_data['info'] = text
        read_json_data['summarizedInfo'] = final_summarize_text

    return json.dumps(read_json_data)
    
        
        
    

        







