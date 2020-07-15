import re
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu
from Text_summary import generate_summary


sentence = "NH Mind is the Business Guide of the organization and is meant to create a common understanding and language for all the NSLHUB leaders. Leaders cannot function effectively unless they understand the common objective, environment and framework. NH Mind is the collective clarity of all the leaders in the organization.NSLHUB is committed to be governed by the dictates of NH Mind. It is NH Mind that shapes the personality of NSLHUB as a whole. NSLHUB believes that the culture of an organization is not by default, but by design. In this context, NSLHUB also believes that it is the NH Mind that would shape the culture of the organization."

new_list = list(re.sub(r'([^\s\w]|_)+', ' ', sentence).split())
result = generate_summary(sentence)
res = json.loads(result)
reference_list = res['summarizedInfo']
summarized_list = list(re.sub(r'([^\s\w]|_)+', ' ', reference_list).split())
summarized_list = [summarized_list]


def test1():
    reference = summarized_list
    candidate = new_list
    score = sentence_bleu(reference, candidate, weights=(1,0,0,0))
    return score


if __name__ == '__main__':
    print(test1())



