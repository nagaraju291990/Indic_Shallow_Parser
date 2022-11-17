import sys
import numpy as np
import tensorflow as tf
import tokenization
import codecs
#from load_models_root import get_root
from typing import List, Optional
import attr
import ast

from indicnlp import common
from indicnlp import loader
from indicnlp.tokenize import indic_tokenize  
from indicnlp.tokenize import sentence_tokenize


INDIC_NLP_RESOURCES="/home/vandan/shallow_parser/Completed/code/indic_nlp_resources/"
common.set_resources_path(INDIC_NLP_RESOURCES)
loader.load()

filter_points=[]

for char in codecs.open('out2.chars.txt.final'):
    char = char.strip()
    filter_points.append(char)


VERSION = 5.0

@attr.dataclass
class shallow:
    text: list
    score: float = 1.0
    code_version: str = VERSION
    error: Optional[str] = None

    def to_dict(self):
        return attr.asdict(self)

    def has_error(self):
        return bool(self.error)

def read_labels(_file):
    labels = []
    for line in codecs.open(_file):
        line = line.strip()
        labels.append(line)
    return labels


labels = {
        'all':{
            'pos':read_labels('labels/allcourse/allcourse_pos_v1.txt'),
            },                                                           
        }
tasks = {
        'all':['pos'],
        }

mlm = {
'ass':'as', 
'ban':'ba', 
'bodo':'bo', 
'guj':'gu', 
'hin':'hi', 
'kan':'ka', 
'kon':'ko', 
'mal':'ma', 
'mar':'mr', 
'nep':'ne', 
'ori':'or', 
'pun':'pu', 
'tam':'ta', 
'tel':'te', 
'urd':'ud', 
}
def load_online_model(lang, model_id):
    print ('Language :', lang)
    print ('/var/vandan/ilci/modelCRF_output_'+mlm[lang]+'_ilciposcourse/saved_model/model-'+model_id)
    model_path = '/var/vandan/ilci/modelCRF_output_'+mlm[lang]+'_ilciposcourse/saved_model/model-'+str(model_id)
    return {'pos':tf.saved_model.load(model_path)}

models = {}


vocab_file = {
        'all':'vocab.txt',
        }
tokenizer = {
        'all': tokenization.FullTokenizer(vocab_file=vocab_file['all'], do_lower_case=True, split_on_punc=False),
        }

processor_text_fn = tokenization.convert_to_unicode


class InputExample(object):
    def __init__(self,
                 guid,
                 text_a,
                 text_b=None,
                 label=None,
                 weight=None,
                 example_id=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.weight = weight
        self.example_id = example_id


def convert_single_example(ex_index, example, max_seq_length, tokenizer):
    print ('here 2', example.text_a)
    tokens_a = tokenizer.tokenize(example.text_a)
    print (tokens_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    seg_id_a = 0
    seg_id_b = 1
    seg_id_cls = 0
    seg_id_pad = 0

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(seg_id_cls)


    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(seg_id_a)

    tokens.append("[SEP]")
    segment_ids.append(seg_id_a)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(seg_id_b)
        tokens.append("[SEP]")
        segment_ids.append(seg_id_b)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(seg_id_pad)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, tokens_a


def pre_common(text, language='hin'):
    example = InputExample(guid="unused_id", text_a=text, text_b=None, label=None)
    sample = convert_single_example(1, example, 400, tokenizer['all'])
    return sample

def predict_common(text, model, label, language='hin'):
    sample = pre_common(text, 'all')
    #print ('--------------')
    #print (sample)
    #print (language)
    #print (tasks[language])
    preds = model([tf.constant([sample[1]]), tf.constant([sample[2]]), tf.constant([sample[0]])])
    #print (preds)

    output = {}
    cc = 0
    for _idx in range(len(tasks['all'])):
        _label=label[tasks['all'][_idx]]
        #print (tasks[language])
        #print (_label)
        #print (preds[_idx][0])
        #print ([i for i in tf.math.argmax(preds[_idx], 1).numpy()])
        #print ([_label[i] for i in tf.math.argmax(preds[_idx], 1).numpy()])
        
        class_index = [_label[i] for i in preds[_idx][0][0].numpy()]
        pred_labels = []
        for i,j in zip(['<START>']+sample[3]+['<END>'], class_index):
            print (i,'---',j)
            if '<START>'==i or '<END>'==i:
                continue
            if '##' not in i:
                if j=='unk':
                    j=''
                pred_labels.append(j)
                cc = cc + 1
            
        output[tasks['all'][_idx]] = [text.split()[i]+'G:$:$:G'+pred_labels[i] for i in range(len(text.split()))]
    return output

def _all(text, language='hin'):
    label = labels['all']
    model = models[language]['pos']
    prediction = predict_common(text, model, label, 'all')
    return prediction


def print_start_sentence(sent_count):
    return '<Sentence id="'+str(sent_count)+'">'

def print_end_sentence():
    return '</Sentence>'
    
r_list = read_labels('r_list.txt')

def shallow_parse(text, language='hin', mode='ssf', model_id='50000'):

    if language not in models:
        models.clear()
        models[language]=load_online_model(language, model_id)


    org_text = text
    text = text.replace('\u200d', '')
    text = text.replace('\u200b', '')
    text = text.replace('\u200d', '')
    text = text.replace('\u200e', '')
    text = " ".join(text.split())

    indic_map = {
            'hin':'hi',
            'kan':'kn',
            'mar':'mr',
            'urd':'ud',
            'tel':'te',
            'mal':'ml',
            'ban':'bn'
            }
    sentences = [text]
    print ('here 1', text)
    output_list = []
    for sentence in sentences:
        p_all = _all(sentence, language)
        output_list.append(p_all)
        return shallow(text=str(output_list))

    
    return send


#text = 'কেন্দ্ৰীয় চৰকাৰে সোনকালেই কেন্দ্ৰীয় কৰ্মচাৰীসকলৰ সূতৰ ধন তেওঁলোকৰ পিএফ একাউণ্টলৈ স্থানান্তৰ কৰিব পাৰে।'
#print (shallow_parse(text, 'ass'))


#text = 'कल पुलिस ने सोनिया गांधी और राहुल गांधी के घरों को घेर लिया था।'
#print (shallow_parse(text, 'hin', '10000'))



