from django.test import TestCase
from django.test import Client
from unittest.mock import patch, Mock
import sys
sys.path.append('../src')


class NLPPreprocessingTest(TestCase):

    def setUp(self):
        self.client= Client()
        
        self.payload= {"text": "At NSLHUB leaders are given oppurtunity to grow by taking end-to-end ownership of their responsibilities"}



    @classmethod
    def setUpClass(cls):
        super(NLPPreprocessingTest, cls).setUpClass()

    def test_get_health(self):
        response= self.client.get('/get_health')

        self.assertEqual(response.status_code, 200)

    
        

    




