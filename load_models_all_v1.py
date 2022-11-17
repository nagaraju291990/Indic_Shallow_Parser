import sys
import numpy as np
import tensorflow as tf
import tokenization
import codecs
from load_models_root import get_root
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

tasks = {
        'ban':['pos', 'chunk'],
        'hin':['pos', 'hi_morph_pos','hi_morph_gender','hi_morph_number','hi_morph_person','hi_morph_case','hi_morph_vib','chunk'],
        'mar':['pos', 'mr_morph_pos','mr_morph_gender','mr_morph_number','mr_morph_person','mr_morph_case','chunk'],
        'kan':['pos', 'ka_morph_pos','ka_morph_gender','ka_morph_number','ka_morph_person','ka_morph_case','chunk'],
        'tel':['pos', 'te_morph_pos','te_morph_gender','te_morph_number','te_morph_person','te_morph_case','chunk'],
        'mal':['pos', 'ml_morph_pos','ml_morph_gender','ml_morph_number','ml_morph_person','ml_morph_case','chunk'],
        'urd':['pos', 'ud_morph_pos','ud_morph_gender','ud_morph_number','ud_morph_person','ud_morph_case','chunk']
        }
labels = {
        'hin':{
            'pos':read_labels('labels/hin/hi_pos_v1.txt'),
            'chunk':read_labels('labels/hin/hi_chunk_v1.txt'),
            'hi_morph_pos':read_labels('labels/hin/hi_morph_pos_v1.txt'),
            'hi_morph_gender':read_labels('labels/hin/hi_morph_gender_v1.txt'),
            'hi_morph_number':read_labels('labels/hin/hi_morph_number_v1.txt'),
            'hi_morph_person':read_labels('labels/hin/hi_morph_person_v1.txt'),
            'hi_morph_case':read_labels('labels/hin/hi_morph_case_v1.txt'),
            'hi_morph_vib':read_labels('labels/hin/hi_morph_vib_v1.txt')
            },
         'mar':{
            'pos':read_labels('labels/mar/mr_pos_v1.txt'),
            'mr_morph_pos':read_labels('labels/mar/mr_morph_pos_v1.txt'),
            'mr_morph_gender':read_labels('labels/mar/mr_morph_gender_v1.txt'),
            'mr_morph_number':read_labels('labels/mar/mr_morph_number_v1.txt'),
            'mr_morph_person':read_labels('labels/mar/mr_morph_person_v1.txt'),
            'mr_morph_case':read_labels('labels/mar/mr_morph_case_v1.txt'),
            'chunk':read_labels('labels/mar/mr_chunk_v1.txt')
            },
          'kan':{
            'pos':read_labels('labels/kan/ka_pos_v1.txt'),
            'ka_morph_pos':read_labels('labels/kan/ka_morph_pos_v1.txt'),
            'ka_morph_gender':read_labels('labels/kan/ka_morph_gender_v1.txt'),
            'ka_morph_number':read_labels('labels/kan/ka_morph_number_v1.txt'),
            'ka_morph_person':read_labels('labels/kan/ka_morph_person_v1.txt'),
            'ka_morph_case':read_labels('labels/kan/ka_morph_case_v1.txt'),
            'chunk':read_labels('labels/kan/ka_chunk_v1.txt'),
            },
          'mal':{
            'pos':read_labels('labels/mal/ml_pos_v1.txt'),
            'ml_morph_pos':read_labels('labels/mal/ml_morph_pos_v1.txt'),
            'ml_morph_gender':read_labels('labels/mal/ml_morph_gender_v1.txt'),
            'ml_morph_number':read_labels('labels/mal/ml_morph_number_v1.txt'),
            'ml_morph_person':read_labels('labels/mal/ml_morph_person_v1.txt'),
            'ml_morph_case':read_labels('labels/mal/ml_morph_case_v1.txt'),
            'chunk':read_labels('labels/mal/ml_chunk_v1.txt'),
            },
          'tel':{
            'pos':read_labels('labels/tel/te_pos_v1.txt'),
            'te_morph_pos':read_labels('labels/tel/te_morph_pos_v1.txt'),
            'te_morph_gender':read_labels('labels/tel/te_morph_gender_v1.txt'),
            'te_morph_number':read_labels('labels/tel/te_morph_number_v1.txt'),
            'te_morph_person':read_labels('labels/tel/te_morph_person_v1.txt'),
            'te_morph_case':read_labels('labels/tel/te_morph_case_v1.txt'),
            'chunk':read_labels('labels/tel/te_chunk_v1.txt'),
            },
          'urd':{
            'pos':read_labels('labels/urd/ud_pos_v1.txt'),
            'ud_morph_pos':read_labels('labels/urd/ud_morph_pos_v1.txt'),
            'ud_morph_gender':read_labels('labels/urd/ud_morph_gender_v1.txt'),
            'ud_morph_number':read_labels('labels/urd/ud_morph_number_v1.txt'),
            'ud_morph_person':read_labels('labels/urd/ud_morph_person_v1.txt'),
            'ud_morph_case':read_labels('labels/urd/ud_morph_case_v1.txt'),
            'chunk':read_labels('labels/urd/ud_chunk_v1.txt'),
            },
          'ban':{
            'pos':read_labels('labels/ban/bn_pos_v1.txt'),
            'chunk':read_labels('labels/ban/bn_chunk_v1.txt'),
            }
        }

models = {
        'hin':{'all':tf.saved_model.load('models/hi_all')},
        'mar':{'all':tf.saved_model.load('models/mr_all')},
        'kan':{'all':tf.saved_model.load('models/ka_all')},
        'mal':{'all':tf.saved_model.load('models/ml_all')},
        'ban':{'all':tf.saved_model.load('models/bn_all')},
        'tel':{'all':tf.saved_model.load('models/te_all')},
        'urd':{'all':tf.saved_model.load('models/ud_all')},
        }

vocab_file = {
        'hin':'models/hi_all/assets/vocab.txt',
        'mar':'models/mr_all/assets/vocab.txt',
        'kan':'models/ka_all/assets/vocab.txt',
        'mal':'models/ml_all/assets/vocab.txt',
        'tel':'models/te_all/assets/vocab.txt',
        'urd':'models/ud_all/assets/vocab.txt',
        'ban':'models/bn_all/assets/vocab.txt',
        }
tokenizer = {
        'hin': tokenization.FullTokenizer(vocab_file=vocab_file['hin'], do_lower_case=True, split_on_punc=False),
        'mar': tokenization.FullTokenizer(vocab_file=vocab_file['mar'], do_lower_case=True, split_on_punc=False),
        'kan': tokenization.FullTokenizer(vocab_file=vocab_file['kan'], do_lower_case=True, split_on_punc=False),
        'mal': tokenization.FullTokenizer(vocab_file=vocab_file['mal'], do_lower_case=True, split_on_punc=False),
        'tel': tokenization.FullTokenizer(vocab_file=vocab_file['tel'], do_lower_case=True, split_on_punc=False),
        'urd': tokenization.FullTokenizer(vocab_file=vocab_file['urd'], do_lower_case=True, split_on_punc=False),
        'ban': tokenization.FullTokenizer(vocab_file=vocab_file['ban'], do_lower_case=True, split_on_punc=False)
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


def pre_common(text, language='hin'):
    example = InputExample(guid="unused_id", text_a=text, text_b=None, label=None)
    sample = convert_single_example(1, example, 400, tokenizer[language])
    return sample

def predict_common(text, model, label, language='hin'):
    sample = pre_common(text, language)
    #print (language)
    #print (tasks[language])
    preds = model([tf.constant([sample[1]]), tf.constant([sample[2]]), tf.constant([sample[0]])])
    output = {}
    cc = 0
    for _idx in range(len(tasks[language])):
        _label=label[tasks[language][_idx]]
        #print (tasks[language])
        #print (_label)
        #print ([i.numpy() for i in tf.math.argmax(preds[_idx], 2)[0]])
        class_index = [_label[i] for i in tf.math.argmax(preds[_idx], 2)[0]]
        pred_labels = []
        for i,j in zip(['<START>']+sample[3]+['<END>'], class_index):
            if '<START>'==i or '<END>'==i:
                continue
            if '##' not in i:
                if j=='unk':
                    j=''
                pred_labels.append(j)
                cc = cc + 1
        output[tasks[language][_idx]] = [text.split()[i]+':'+pred_labels[i] for i in range(len(text.split()))]
    return output

def _all(text, language='hin'):
    label = labels[language]
    model = models[language]['all']
    prediction = predict_common(text, model, label, language)
    return prediction


def print_start_sentence(sent_count):
    return '<Sentence id="'+str(sent_count)+'">'

def print_end_sentence():
    return '</Sentence>'
    
r_list = read_labels('r_list.txt')

def shallow_parse(text, language='hin', mode='ssf'):
    org_text = text
    text = text.replace('\u200d', '')
    text = text.replace('\u200b', '')
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
    indic_map = {
            'hin':'hi',
            'kan':'kn',
            'mar':'mr',
            'urd':'ud',
            'tel':'te',
            'mal':'ml',
            'ban':'bn'
            }
    sentences=sentence_tokenize.sentence_split(text, lang=indic_map[language])
    #print (sentences)
    sent_count = 0
    output = []
    output_list = []
    for sentence in sentences:
        sent_count = sent_count + 1
        if language=='hin':
            morphroot = get_root(sentence)
        else:
            morphroot = [w+':'+w for w in sentence.split()]
        p_all = _all(sentence, language)
        if len(tasks[language])==2:
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


        p_all['root']=morphroot
        output_list.append(p_all)

        out = print_start_sentence(sent_count)
        output.append(out)
        flag = False
        count = 0
        c = 0
        for idx in range(len(sentence.split())):
            if 'B-' in chunk_[idx].split(':')[1] or idx==0:
                if flag:
                    out = "\t))"
                    output.append(out)
                count = count + 1
                c = 1

                out = str(count)+"\t((\t"+chunk_[idx].split(':')[1].replace('B-','')
                if idx==0 and 'I-' in chunk_[idx].split(':')[1]:
                    out = out.replace('I-','')
                output.append(out)
                if len(tasks[language])==2:
                    out = str(count)+'.'+str(c)+"\t"+sentence.split()[idx]+'\t'+pos_[idx].split(':')[1]+'\t'+"<fs af='"+morphroot[idx].split(':')[1]+','+','+','+','+','+','+','+"'>"
                elif len(tasks[language])==8:
                    out = str(count)+'.'+str(c)+"\t"+sentence.split()[idx]+'\t'+pos_[idx].split(':')[1]+'\t'+"<fs af='"+morphroot[idx].split(':')[1]+','+morphpos[idx].split(':')[1]+','+morphgender[idx].split(':')[1]+','+morphnumber[idx].split(':')[1]+','+morphperson[idx].split(':')[1]+','+morphcase[idx].split(':')[1]+','+ morphvib[idx].split(':')[1]+','+morphvib[idx].split(':')[1]+"'>"
                else:
                    out = str(count)+'.'+str(c)+"\t"+sentence.split()[idx]+'\t'+pos_[idx].split(':')[1]+'\t'+"<fs af='"+morphroot[idx].split(':')[1]+','+morphpos[idx].split(':')[1]+','+morphgender[idx].split(':')[1]+','+morphnumber[idx].split(':')[1]+','+morphperson[idx].split(':')[1]+','+morphcase[idx].split(':')[1]+','+','+"'>"

                output.append(out)
                flag = True

            elif 'I-' in chunk_[idx].split(':')[1]:
                c = c+1
                if len(tasks[language])==2:
                    out = str(count)+'.'+str(c)+"\t"+sentence.split()[idx]+'\t'+pos_[idx].split(':')[1]+'\t'+"<fs af='"+morphroot[idx].split(':')[1]+','+','+','+','+','+','+"'>"
                elif len(tasks[language])==8:
                    out = str(count)+'.'+str(c)+"\t"+sentence.split()[idx]+'\t'+pos_[idx].split(':')[1]+'\t'+"<fs af='"+morphroot[idx].split(':')[1]+','+morphpos[idx].split(':')[1]+','+morphgender[idx].split(':')[1]+','+morphnumber[idx].split(':')[1]+','+morphperson[idx].split(':')[1]+','+morphcase[idx].split(':')[1]+','+ morphvib[idx].split(':')[1]+','+morphvib[idx].split(':')[1]+"'>"
                else:
                    out = str(count)+'.'+str(c)+"\t"+sentence.split()[idx]+'\t'+pos_[idx].split(':')[1]+'\t'+"<fs af='"+morphroot[idx].split(':')[1]+','+morphpos[idx].split(':')[1]+','+morphgender[idx].split(':')[1]+','+morphnumber[idx].split(':')[1]+','+morphperson[idx].split(':')[1]+','+morphcase[idx].split(':')[1]+','+','+"'>"

                output.append(out)


        out = "\t))"
        output.append(out)
        out = print_end_sentence()
        output.append(out)
        send = shallow(text=output)

    if mode=='list':
        return shallow(text=str(output_list))

    
    return send


text = "ट्रेन में होने वाले हर अपराध में इस गाँव का कोई न कोई शामिल मिल जाएगा ।"
print (shallow_parse(text, 'hin'))
text = "त्यामुळे याबाबत निर्णय आता निवडणूक आयोग घेणार आहे."
print (shallow_parse(text, 'mar'))
text = 'ಬೆಂಗಳೂರು ಮೆಟ್ರೋದ ಬಹುನಿರೀಕ್ಷಿತ ಪರ್ಪಲ್ ಲೈನ್ ವಿಸ್ತರಣೆಯ ಬೈಯಪ್ಪನಹಳ್ಳಿಯಿಂದ ವೈಟ್‌ಫೀಲ್ಡ್‌ಗೆ ಸೆಪ್ಟೆಂಬರ್‌ನಲ್ಲಿ ಪ್ರಾಯೋಗಿಕ ಚಾಲನೆ ಪ್ರಾರಂಭವಾಗಲಿವೆ ಎಂದು ಬೆಂಗಳೂರು ಮೆಟ್ರೋ ರೈಲು ನಿಗಮದ ಹಿರಿಯ ಅಧಿಕಾರಿಗಳು ತಿಳಿಸಿದ್ದಾರೆ.'
print (shallow_parse(text, 'kan'))
text = 'അതേസമയം സോണിയാ ഗാന്ധിയേയും രാഹുൽ ഗാന്ധിയേയും ഇ.ഡിയെ ഉപയോഗിച്ച് കേന്ദ്ര സർക്കാർ കള്ളക്കേസിൽ കുടുക്കാൻ ശ്രമിക്കുന്നുവെന്നു ആരോപിച്ച്'
print (shallow_parse(text, 'mal'))
text = 'حال ہی میں دیشا پٹانی نے ایک انٹرویو میں اپنے کرش کا نام بتایا ہے۔ اداکارہ نے کہا کہ اسکول کے زمانے میں وہ بالی ووڈ اداکار رنبیر کپور کی بہت زیادہ دیوانی تھی اور ان کی وجہ سے مجھے کئی بار حادثات کا سامنا بھی کرنا پڑا۔ اداکارہ کا کہنا تھا کہ ان کے شہر میں رنبیر کپور کا ایک بڑا پوسٹر تھا جسے وہ ہر وقت وہاں سے آتے جاتے دیکھتے رہتے تھے اور اس سلسلے میں وہ اکثر سڑک پر کسی نہ کسی چیز سے ٹکرا جاتی تھیں۔'
print (shallow_parse(text, 'urd'))
text = 'ఈ సందర్భంగా సైనికుల స్పూర్తితో వెండితెరపై అలరించిన తెలుగు హీరోలపై ఫోకస్.'
print (shallow_parse(text, 'tel'))
text = 'এইমস-এ করা পার্থ চট্টোপাধ্যায়ের মেডিক্যাল রিপোর্টে কী কী বেরিয়েছে?'
print (shallow_parse(text, 'ban'))

#for line in shallow_parse(text):
#    print (line)
