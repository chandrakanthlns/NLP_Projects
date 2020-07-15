from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import sys
import json
sys.path.append('../src')
from django.views.decorators.csrf import csrf_exempt
# from src.helper_functions import pos_tagger,ner_tagger,coreferece_resolution
from main import final_test
from util_functions import create_logger
logger= create_logger()


def get_health(request):

    if (request.method == 'GET'):
        return HttpResponse(json.dumps({"message": "NLP preprocessing service is running fine"}), content_type= 'application/json')

@csrf_exempt
def nlp_prep(request):

    try:
        if (request.method == 'POST'):
            data= json.loads(request.body)
            nlp_prepresponse= final_test(data['text'])
            for resp in nlp_prepresponse:
                return HttpResponse(json.dumps(resp), content_type= 'application/json')
                      
    except Exception as e:
        logger.info("exception occured %s",e)

    # return HttpResponse(json.dumps(nlp_prepresponse), content_type= 'application/json')


