import sys
import numpy as np
import tensorflow as tf
import tokenization
import codecs
#from load_models_root import get_root
from typing import List, Optional
import attr
import ast
import json
from indicnlp import common
from indicnlp import loader
from indicnlp.tokenize import indic_tokenize  
from indicnlp.tokenize import sentence_tokenize

_in_hist = {}

for line in codecs.open('oct_24_2022_history.json','r'):
    tmp = json.loads(line)
    #print (tmp.keys())
    _in_hist[list(tmp.keys())[0]]=tmp[list(tmp.keys())[0]]

#for i in _in_hist:
#    print (i, _in_hist[i])
print ('Total existing :', len(_in_hist))

def get_root(sentences):
    o = []
    for _s in sentences:
        s=_s.split()
        o.append([i+':'+i for i in s])

    return o

INDIC_NLP_RESOURCES="indic_nlp_resources"
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


tasks = {
        'hin':['pos', 'hi_morph_pos','hi_morph_gender','hi_morph_number','hi_morph_person','hi_morph_case','hi_morph_vib','chunk'],
        'mal':['pos', 'ml_morph_pos','ml_morph_gender','ml_morph_number','ml_morph_person','ml_morph_case','chunk'],
        }

labels = {
        'hin':{
            'pos':read_labels('labels/hin/hi_pos_v2.txt'),
            'chunk':read_labels('labels/hin/hi_chunk_v2.txt'),
            'morph_pos':read_labels('labels/hin/hi_morph_pos_v2.txt'),
            'morph_gender':read_labels('labels/hin/hi_morph_gender_v2.txt'),
            'morph_number':read_labels('labels/hin/hi_morph_number_v2.txt'),
            'morph_person':read_labels('labels/hin/hi_morph_person_v2.txt'),
            'morph_case':read_labels('labels/hin/hi_morph_case_v2.txt'),
            'morph_vib':read_labels('labels/hin/hi_morph_vib_v2.txt')
        },
       'mal':{
            'pos':read_labels('labels/mal/ml_pos_v2.txt'),
            'chunk':read_labels('labels/mal/ml_chunk_v2.txt'),
            'morph_pos':read_labels('labels/mal/ml_morph_pos_v2.txt'),
            'morph_gender':read_labels('labels/mal/ml_morph_gender_v2.txt'),
            'morph_number':read_labels('labels/mal/ml_morph_number_v2.txt'),
            'morph_person':read_labels('labels/mal/ml_morph_person_v2.txt'),
            'morph_case':read_labels('labels/mal/ml_morph_case_v2.txt'),
        }
       
    }

models = {
        'hin': {'all':tf.saved_model.load('models/hi_all')},
        'mal': {'all':tf.saved_model.load('models/ml_all')},
    }

print (labels)
vocab_file = 'vocab.txt'
tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True, split_on_punc=False)
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
    tokens_a = tokenizer.tokenize(example.text_a)
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


def pre_common(text):
    example = InputExample(guid="unused_id", text_a=text, text_b=None, label=None)
    sample = convert_single_example(1, example, 400, tokenizer)
    return sample


def predict_common(text, model, label):
    print ('in :', text)

    final_output = {}
    sample = []
    _text = []
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    for s in text:
        _text.append(s)
        _ = pre_common(s)
        a1.append(_[1])
        a2.append(_[2])
        a3.append(_[0])
        a4.append(_[3])

    preds = model([tf.constant(a1), tf.constant(a2), tf.constant(a3)])
    s = 0
    for sample in a4:
        cc = 0
        output = {}
        for _idx in range(len(tasks)):
            label=labels[tasks[_idx]]
            print (tasks[_idx],label)
            print ([i for i in preds[_idx][0][s].numpy()])
            class_index = [label[i] for i in preds[_idx][0][s].numpy()]
            pred_labels = []

            for i,j in zip(['<START>']+sample+['<END>'], class_index):
                print (i,'=========',j)
                if '<START>'==i or '<END>'==i:
                    continue
                if '##' not in i:
                    if j=='unk':
                        j=''
                    print (i,'====ccccc=====',j)
                    pred_labels.append(j)
                    cc = cc + 1
            print (text[s],'\n===\n',sample)
            print (len(text[s].split()),'\n===\n',len(sample),'\n====',len(pred_labels)) 
            if len(text[s].split()) == len(pred_labels):
                output[tasks[_idx]] = [text[s].split()[i]+'$%:%$'+pred_labels[i] for i in range(len(text[s].split()))]
            else:
                output[tasks[_idx]] = [text[s].split()[i]+'$%:%$'+' ' for i in range(len(text[s].split()))]

        final_output[text[s]] = output

        s = s + 1
    return final_output

def _all(text):
    label = labels
    model = models['all']
    prediction = predict_common(text, model, label)
    return prediction


def print_start_sentence(sent_count):
    #print ('<Sentence id="'+str(sent_count)+'">')
    return '<Sentence id="'+str(sent_count)+'">'

def print_end_sentence():
    #print ('</Sentence>')
    return '</Sentence>'
r_list = read_labels('r_list.txt')


def shallow_parse(text, mode='ssf'):
    org_text = text
    
    if type(text)==str:
        text = text.replace('\u200d', '')
        text = text.replace('\u200b', '')
        text=text.replace("व ें","वें")
        text=text.replace("वें"," वें")
        text=text.replace("स े"," से ")
        text = text.replace("म ें","में")
        text = text.replace("म ें"," में")
        text=text.replace("में"," में")
        text=text.replace("के"," के")
        text=text.replace("16 ़"," 16")
        text = text.replace('\u200d', '')
        text = text.replace('\u200e', '')
        text = " ".join(text.split())
        _text  = []
        for t in text.split():
            if t not in r_list:
                _text.append(t)
        text = " ".join(_text)
        text = " ".join(indic_tokenize.trivial_tokenize(text))

        flag = False
        text_f = []
        for word in text.split():
            if word in filter_points:
                pass
            else:
                text_f.append(word)
        text = " ".join(text_f)

        #print ('Text1 : ', text)
        #print ('Text2 : ', text.split())
        #sentences = [text]#indic_tokenizer.tokenize_text(text)
        sentences=sentence_tokenize.sentence_split(text, lang='hi')

    else:
        sentences = []
        for _s in text:
            text = _s.replace('\u200d', '')
            text = text.replace('\u200b', '')
            text = text.replace('\u200d', '')
            text = text.replace('\u200e', '')


            text = text.replace('\u200d', '')
            text = text.replace('\u200b', '')
            text=text.replace("व ें","वें")
            text=text.replace("वें"," वें")
            text=text.replace("स े"," से ")
            text = text.replace("म ें","में")
            text = text.replace("म ें"," में")
            text=text.replace("में"," में")
            text=text.replace("के"," के")
            text=text.replace("16 ़"," 16")
            text = text.replace('\u200d', '')
            text = text.replace('\u200e', '')
            text = " ".join(text.split())
            _text  = []
            for t in text.split():
                if t not in r_list:
                    _text.append(t)
            text = " ".join(_text)
            text = " ".join(indic_tokenize.trivial_tokenize(text))


            text = " ".join(text.split())

            text_f = []
            for word in text.split():
                if word in filter_points:
                    pass
                else:
                    text_f.append(word)
            text = " ".join(text_f)





            text = " ".join(indic_tokenize.trivial_tokenize(text))
            sentences.append(text)
    _sentences = []
    for sent in sentences:
        print (sent)
        if sent not in _in_hist:
            print ('\t NO')
            _sentences.append(sent)
        else:
            print ('\t YES')


    sent_count = 0
    output = []
    output_list = []
    #for sentence in sentences:
    if len(_sentences)>0:
        histf = open('oct_24_2022_history.json','a')
        sent_count = sent_count + 1
        #sentence = indic_tokenizer.tokenize(sentence)
        morphroot = get_root(_sentences)
        p_all = _all(_sentences)
        #print (json.dumps(p_all))
        _sid = 0
        for sentence in _sentences:
            chunk_ = p_all[sentence][tasks[7]]
            pos_ = p_all[sentence][tasks[0]]
            morphpos = p_all[sentence][tasks[1]]
            morphgender = p_all[sentence][tasks[2]]
            morphnumber = p_all[sentence][tasks[3]]
            morphperson = p_all[sentence][tasks[4]]
            morphcase = p_all[sentence][tasks[5]]
            morphvib = p_all[sentence][tasks[6]]
            p_all[sentence]['root']=morphroot[_sid]

            _in_hist[sentence]=p_all[sentence]

            histf.write(json.dumps({sentence:p_all[sentence]})+'\n')
            #output_list.append(p_all[sentence])

            _sid = _sid + 1
        histf.close()

    for sent in sentences:
        output_list.append(_in_hist[sent])
    
    return output_list


text = "ट्रेन में होने वाले हर अपराध में इस गाँव का कोई न कोई शामिल मिल जाएगा ।"

text = "ट्रेन में होने वाले हर अपराध में इस गाँव का कोई न कोई शामिल मिल जाएगा । कार्यक्रम में देशभर के केंद्रीय मंत्री भी शामिल होंगे।"

#line = shallow_parse(text)
#print (line)
