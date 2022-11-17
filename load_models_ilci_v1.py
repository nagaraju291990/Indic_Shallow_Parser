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


INDIC_NLP_RESOURCES="/home/vandan/work/research/tools/shallow_parser/indic_nlp_resources/"
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
        'ass':{
            'pos':read_labels('labels/ass/ass_pos_v1.txt'),
            },
        'ban':{
            'pos':read_labels('labels/ban/ban_pos_v1.txt'),
            },            
        'bodo':{
            'pos':read_labels('labels/bodo/bodo_pos_v1.txt'),
            },            
        'guj':{
            'pos':read_labels('labels/guj/guj_pos_v1.txt'),
            },
        'hin':{
            'pos':read_labels('labels/hin/hin_pos_v1.txt'),
            },            
        'kan':{
            'pos':read_labels('labels/kan/kan_pos_v1.txt'),
            },            
        'kon':{
            'pos':read_labels('labels/kon/kon_pos_v1.txt'),
            },            
        'mal':{
            'pos':read_labels('labels/mal/mal_pos_v1.txt'),
            },            
        'mar':{
            'pos':read_labels('labels/mar/mar_pos_v1.txt'),
            },            
        'nep':{
            'pos':read_labels('labels/nep/nep_pos_v1.txt'),
            },            
        'ori':{
            'pos':read_labels('labels/ori/ori_pos_v1.txt'),
            }, 
        'pun':{
            'pos':read_labels('labels/pun/pun_pos_v1.txt'),
            }, 
        'tam':{
            'pos':read_labels('labels/tam/tam_pos_v1.txt'),
            }, 
        'tel':{
            'pos':read_labels('labels/tel/tel_pos_v1.txt'),
            }, 
        'urd':{
            'pos':read_labels('labels/urd/urd_pos_v1.txt'),
            }, 
        'all':{
            'pos':read_labels('labels/all/all_pos_v1.txt'),
            },
        'all2':{
            'pos':read_labels('labels/all2/all2_pos_v1.txt'),
            },            
        'allND':{
            'pos':read_labels('labels/allND/allND_pos_v1.txt'),
            },
        'indoaryan':{
            'pos':read_labels('labels/indoaryan/indoaryan_pos_v1.txt'),
            },
        'indoaryan2':{
            'pos':read_labels('labels/indoaryan2/indoaryan2_pos_v1.txt'),
            },            
        'dravidian':{
            'pos':read_labels('labels/dravidian/dravidian_pos_v1.txt'),
            }, 
        'dravidian2':{
            'pos':read_labels('labels/dravidian2/dravidian2_pos_v1.txt'),
            },              
        'dravidianplus':{
            'pos':read_labels('labels/dravidianplus/dravidianplus_pos_v1.txt'),
            },                                                           
        }
tasks = {
        'ass':['pos'],
        'ban':['pos'],
        'bodo':['pos'],
        'guj':['pos'],
        'hin':['pos'],
        'kan':['pos'],
        'kon':['pos'],
        'mal':['pos'],
        'mar':['pos'],
        'nep':['pos'],
        'ori':['pos'],
        'pun':['pos'],
        'tam':['pos'],
        'tel':['pos'],
        'urd':['pos'],
        'all':['pos'],
        'all2':['pos'],
        'allND':['pos'],
        'indoaryan':['pos'],
        'indoaryan2':['pos'],
        'dravidian':['pos'],  
        'dravidian2':['pos'],  
        'dravidianplus':['pos'],        
        }

def load_online_model(lang):
    print ('Language :', lang)
    print ('models/'+lang+'_pos')
    return {'pos':tf.saved_model.load('models/'+lang+'_pos')}

models = {}

#modelsi = {
#        'ass':{'pos':tf.saved_model.load('models/ass_pos')},
#        'ban':{'pos':tf.saved_model.load('models/ban_pos')},
#        'bodo':{'pos':tf.saved_model.load('models/bodo_pos')},
#        'guj':{'pos':tf.saved_model.load('models/guj_pos')},
#        'hin':{'pos':tf.saved_model.load('models/hin_pos')},        
#        'kan':{'pos':tf.saved_model.load('models/kan_pos')},
#        'kon':{'pos':tf.saved_model.load('models/kon_pos')},        
#        'mal':{'pos':tf.saved_model.load('models/mal_pos')},
#        'mar':{'pos':tf.saved_model.load('models/mar_pos')},        
#        'nep':{'pos':tf.saved_model.load('models/nep_pos')},
#        'ori':{'pos':tf.saved_model.load('models/ori_pos')},        
#        'pun':{'pos':tf.saved_model.load('models/pun_pos')},
#        'tam':{'pos':tf.saved_model.load('models/tam_pos')},        
#        'tel':{'pos':tf.saved_model.load('models/tel_pos')},
#        'urd':{'pos':tf.saved_model.load('models/urd_pos')},
#        }

vocab_file = {
        'ass':'models/ass_pos/assets/vocab.txt',
        'ban':'models/ban_pos/assets/vocab.txt',
        'bodo':'models/bodo_pos/assets/vocab.txt',
        'guj':'models/guj_pos/assets/vocab.txt',
        'hin':'models/hin_pos/assets/vocab.txt',
        'kan':'models/kan_pos/assets/vocab.txt',
        'kon':'models/kon_pos/assets/vocab.txt',        
        'mal':'models/mal_pos/assets/vocab.txt',
        'mar':'models/mar_pos/assets/vocab.txt',        
        'nep':'models/nep_pos/assets/vocab.txt',
        'ori':'models/ori_pos/assets/vocab.txt',        
        'pun':'models/pun_pos/assets/vocab.txt',
        'tam':'models/tam_pos/assets/vocab.txt',        
        'tel':'models/tel_pos/assets/vocab.txt',
        'urd':'models/urd_pos/assets/vocab.txt',  
        'all':'models/all_pos/assets/vocab.txt', 
        'all2':'models/all2_pos/assets/vocab.txt', 
        'allND':'models/allND_pos/assets/vocab.txt', 
        'indoaryan':'models/indoaryan_pos/assets/vocab.txt', 
        'indoaryan2':'models/indoaryan2_pos/assets/vocab.txt', 
        'dravidian':'models/dravidian_pos/assets/vocab.txt', 
        'dravidian2':'models/dravidian2_pos/assets/vocab.txt', 
        'dravidianplus':'models/dravidianplus_pos/assets/vocab.txt', 
        }
tokenizer = {
        'ass': tokenization.FullTokenizer(vocab_file=vocab_file['ass'], do_lower_case=True, split_on_punc=False),
        'ban': tokenization.FullTokenizer(vocab_file=vocab_file['ban'], do_lower_case=True, split_on_punc=False),
        'bodo': tokenization.FullTokenizer(vocab_file=vocab_file['bodo'], do_lower_case=True, split_on_punc=False),
        'guj': tokenization.FullTokenizer(vocab_file=vocab_file['guj'], do_lower_case=True, split_on_punc=False),
        'hin': tokenization.FullTokenizer(vocab_file=vocab_file['hin'], do_lower_case=True, split_on_punc=False),
        'kan': tokenization.FullTokenizer(vocab_file=vocab_file['kan'], do_lower_case=True, split_on_punc=False),
        'kon': tokenization.FullTokenizer(vocab_file=vocab_file['kon'], do_lower_case=True, split_on_punc=False),
        'mal': tokenization.FullTokenizer(vocab_file=vocab_file['mal'], do_lower_case=True, split_on_punc=False),
        'mar': tokenization.FullTokenizer(vocab_file=vocab_file['mar'], do_lower_case=True, split_on_punc=False),
        'nep': tokenization.FullTokenizer(vocab_file=vocab_file['nep'], do_lower_case=True, split_on_punc=False),
        'ori': tokenization.FullTokenizer(vocab_file=vocab_file['ori'], do_lower_case=True, split_on_punc=False),
        'pun': tokenization.FullTokenizer(vocab_file=vocab_file['pun'], do_lower_case=True, split_on_punc=False),
        'tam': tokenization.FullTokenizer(vocab_file=vocab_file['tam'], do_lower_case=True, split_on_punc=False),        
        'tel': tokenization.FullTokenizer(vocab_file=vocab_file['tel'], do_lower_case=True, split_on_punc=False),
        'urd': tokenization.FullTokenizer(vocab_file=vocab_file['urd'], do_lower_case=True, split_on_punc=False),
        'all': tokenization.FullTokenizer(vocab_file=vocab_file['all'], do_lower_case=True, split_on_punc=False),        
        'all2': tokenization.FullTokenizer(vocab_file=vocab_file['all2'], do_lower_case=True, split_on_punc=False),
        'allND': tokenization.FullTokenizer(vocab_file=vocab_file['allND'], do_lower_case=True, split_on_punc=False), 
        'indoaryan': tokenization.FullTokenizer(vocab_file=vocab_file['indoaryan'], do_lower_case=True, split_on_punc=False),        
        'indoaryan2': tokenization.FullTokenizer(vocab_file=vocab_file['indoaryan2'], do_lower_case=True, split_on_punc=False),        
        'dravidian': tokenization.FullTokenizer(vocab_file=vocab_file['dravidian'], do_lower_case=True, split_on_punc=False),
        'dravidian2': tokenization.FullTokenizer(vocab_file=vocab_file['dravidian2'], do_lower_case=True, split_on_punc=False),
        'dravidianplus': tokenization.FullTokenizer(vocab_file=vocab_file['dravidianplus'], do_lower_case=True, split_on_punc=False),        
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
    sample = convert_single_example(1, example, 400, tokenizer[language])
    return sample

def predict_common(text, model, label, language='hin'):
    sample = pre_common(text, language)
    #print ('--------------')
    #print (sample)
    #print (language)
    #print (tasks[language])
    preds = model([tf.constant([sample[1]]), tf.constant([sample[2]]), tf.constant([sample[0]])])
    output = {}
    cc = 0
    for _idx in range(len(tasks[language])):
        _label=label[tasks[language][_idx]]
        #print (tasks[language])
        #print (_label)
        #print (preds[_idx])
        #print ([i for i in tf.math.argmax(preds[_idx], 1).numpy()])
        #print ([_label[i] for i in tf.math.argmax(preds[_idx], 1).numpy()])
        class_index = [_label[i] for i in tf.math.argmax(preds[_idx], 1).numpy()]
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
            
        output[tasks[language][_idx]] = [text.split()[i]+'G:$:$:G'+pred_labels[i] for i in range(len(text.split()))]
    return output

def _all(text, language='hin'):
    label = labels[language]
    model = models[language]['pos']
    prediction = predict_common(text, model, label, language)
    return prediction


def print_start_sentence(sent_count):
    return '<Sentence id="'+str(sent_count)+'">'

def print_end_sentence():
    return '</Sentence>'
    
r_list = read_labels('r_list.txt')

def shallow_parse(text, language='hin', mode='ssf'):

    if language not in models:
        models.clear()
        models[language]=load_online_model(language)


    org_text = text
    text = text.replace('\u200d', '')
    text = text.replace('\u200b', '')
    text = text.replace('\u200d', '')
    text = text.replace('\u200e', '')
    text = " ".join(text.split())
    #_text  = []
    #for t in text.split():
    #    if t not in r_list:
    #        _text.append(t)
    #text = " ".join(_text)
    #print (text)
    #text = " ".join(indic_tokenize.trivial_tokenize(text))

    flag = False
    #text_f = []
    #for word in text.split():
    #    if word in filter_points:
    #        pass
    #    else:
    #        text_f.append(word)
    #text = " ".join(text_f)
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
    #sentences=sentence_tokenize.sentence_split(text, lang=indic_map[language])
    #print (sentences)
    sent_count = 0
    output = []
    output_list = []
    for sentence in sentences:
        sent_count = sent_count + 1
        if language=='hin':
            #morphroot = get_root(sentence)
            morphroot = [w+':'+w for w in sentence.split()]
        else:
            morphroot = [w+':'+w for w in sentence.split()]
        p_all = _all(sentence, language)
        if len(tasks[language])==1:
            pos_ = p_all[tasks[language][0]]
        elif len(tasks[language])==2:
            pos_ = p_all[tasks[language][0]]
            chunk_ = p_all[tasks[language][1]]
        else:
            if len(tasks[language])==8:
                chunk_ = p_all[tasks[language][7]]
            else:
                chunk_ = p_all[tasks[language][6]]

            pos_ = p_all[tasks[language][0]]
            morphpos = p_all[tasks[language][1]]
            morphgender = p_all[tasks[language][2]]
            morphnumber = p_all[tasks[language][3]]
            morphperson = p_all[tasks[language][4]]
            morphcase = p_all[tasks[language][5]]
            if len(tasks[language])==8:
                morphvib = p_all[tasks[language][6]]


        output_list.append(p_all)

        return shallow(text=str(output_list))

    
    return send


#text = 'কেন্দ্ৰীয় চৰকাৰে সোনকালেই কেন্দ্ৰীয় কৰ্মচাৰীসকলৰ সূতৰ ধন তেওঁলোকৰ পিএফ একাউণ্টলৈ স্থানান্তৰ কৰিব পাৰে।'
#print (shallow_parse(text, 'ass'))


#text = 'कल पुलिस ने सोनिया गांधी और राहुल गांधी के घरों को घेर लिया था।'
#print (shallow_parse(text, 'hin'))



