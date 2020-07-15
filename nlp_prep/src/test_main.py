import unittest
from unittest import TestCase
import sys
# sys.path.append('..')

# from ..src.helper_functions import  pos_tagger

# sys.path.append('./src')
# from src.helper_functions import pos_tagger,ner_tagger,coreferece_resolution
from helper_functions import pos_tagger, ner_tagger
from util_functions import exception , create_logger, timer

class NLPPreprocessingTests(TestCase):
    
    def test_pos_tagger(self):
        
        pos = pos_tagger("At NSLHUB leaders are given oppurtunity to grow by taking end-to-end ownership of their responsibilities.It involves dealing with TDC-thinking,doing,collaborating.")
        
        expected = 'IN'
        self.assertEqual(pos[0]['pos'] , expected)
        
    def test_ner_tagger(self):
        ner= ner_tagger("At NSLHUB leaders are given oppurtunity to grow by taking end-to-end ownership of their responsibilities.It involves dealing with TDC-thinking,doing,collaborating.")
        expected= 'ORGANIZATION'
        self.assertEqual(ner[0]['ner'], expected)
        
    # def test_coreference_resolution(self):
    #     coref= coreferece_resolution("At NSLHUB leaders are given oppurtunity to grow by taking end-to-end ownership of their responsibilities.It involves dealing with TDC-thinking,doing,collaborating.")
    #     expected= "At NSLHUB leaders are given oppurtunity to grow by taking end - to - end ownership of NSLHUB leadersresponsibilities.It involves dealing with TDC - thinking, doing, collaborating"
    #     self.assertEqual(coref, expected)
        
    




if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)



