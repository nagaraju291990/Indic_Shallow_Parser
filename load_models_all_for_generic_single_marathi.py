import sys
import numpy as np
import tensorflow as tf
import tokenization
import codecs
from ilstokenizer import tokenizer as indic_tokenizer
from load_models_root import get_root


def read_labels(_file):
    labels = []
    for line in codecs.open(_file):
        line = line.strip()
        labels.append(line)
    return labels

def read_files(_file):
    data = []
    for line in codecs.open(_file):
        line = line.strip().split('\t')
        data.append(line)
    return data

tasks = ['pos','chunk']

tests = ["mr_POS_news_v1_test.bio","mr_Chunk_news_v1_test.bio"]


labels = {
    'pos':read_labels('labels_marathi/mr_pos_v1.txt'),
    'chunk':read_labels('labels_marathi/mr_chunk_v1.txt'),
}


models = {
    'pos':tf.saved_model.load('models_marathi/mr_pos'),
    'chunk':tf.saved_model.load('models_marathi/mr_chunk'),
}


vocab_file = 'models_marathi/mr_chunk/assets/vocab.txt'
tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)
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





def predict_common2(text, model, label):
    sample = pre_common(text)
    preds = model([tf.constant([sample[1]]), tf.constant([sample[2]]), tf.constant([sample[0]])])
    output = {}
    for _idx in range(len(tasks)):
        label=labels[tasks[_idx]]
        class_index = [label[i] for i in tf.math.argmax(preds[_idx], 2)[0]]
        pred_labels = []
        for i,j in zip(['<START>']+sample[3]+['<END>'], class_index):
            if '<START>'==i or '<END>'==i:
                continue
            if '##' not in i:
                if j=='unk':
                    j=''

                pred_labels.append(j)
        #print (tasks[_idx], text.split())
        #for i in range(len(text.split())):
        #    print (text.split()[i]+':'+pred_labels[i])

        output[tasks[_idx]] = [text.split()[i]+'@#@'+pred_labels[i] for i in range(len(text.split()))]
    return output


def predict_common(text, model, label):
    sample = pre_common(text)
    preds = model([tf.constant([sample[1]]), tf.constant([sample[2]]), tf.constant([sample[0]])])
    class_index = [label[i] for i in tf.math.argmax(preds, 2)[0]]
    pred_labels = []
    for i,j in zip(['<START>']+sample[3]+['<END>'], class_index):
        if '<START>'==i or '<END>'==i:
            continue
        if '##' not in i:
            if j=='unk':
                j=''

            pred_labels.append(j)

    return [text.split()[i]+'@#@'+pred_labels[i] for i in range(len(text.split()))]




def pos(text):
    label = labels['pos']
    model = models['pos']
    prediction = predict_common(text, model, label)
    return prediction

def chunk(text):
    label = labels['chunk']
    model = models['chunk']
    prediction = predict_common(text, model, label)
    return prediction

def morph_pos(text):
    label = labels['hi_morph_pos']
    model = models['hi_morph_pos']
    prediction = predict_common(text, model, label)
    return prediction

def morph_gender(text):
    label = labels['hi_morph_gender']
    model = models['hi_morph_gender']
    prediction = predict_common(text, model, label)
    return prediction

def morph_number(text):
    label = labels['hi_morph_number']
    model = models['hi_morph_number']
    prediction = predict_common(text, model, label)
    return prediction

def morph_person(text):
    label = labels['hi_morph_person']
    model = models['hi_morph_person']
    prediction = predict_common(text, model, label)
    return prediction

def morph_case(text):
    label = labels['hi_morph_case']
    model = models['hi_morph_case']
    prediction = predict_common(text, model, label)
    return prediction

def morph_vib(text):
    label = labels['hi_morph_vib']
    model = models['hi_morph_vib']
    prediction = predict_common(text, model, label)
    return prediction


def shallow_parse(sentence):
    sentence = indic_tokenizer.tokenize(sentence)
    print (sentence)




#def _all(text):
#    label = labels
#    model = models['all']
#    prediction = predict_common(text, model, label)
#    return prediction


def print_start_sentence(sent_count):
    #print ('<Sentence id="'+str(sent_count)+'">')
    return '<Sentence id="'+str(sent_count)+'">'

def print_end_sentence():
    #print ('</Sentence>')
    return '</Sentence>'

def shallow_parse(text):
    sentences = [text]#indic_tokenizer.tokenize_text(text)
    sent_count = 0
    output = {}
    for sentence in sentences:
        sent_count = sent_count + 1
        #sentence = indic_tokenizer.tokenize(sentence)
        morphroot = get_root(sentence)
        #p_all = _all(sentence)
        chunk_ = chunk(sentence)
        pos_ = pos(sentence)
        
        #morphpos = morph_pos(sentence)
        #morphgender = morph_gender(sentence)
        #morphnumber = morph_number(sentence)
        #morphperson = morph_person(sentence)
        #morphcase = morph_case(sentence)
        #morphvib = morph_vib(sentence)
        output[tasks[0]] = pos_
        #output[tasks[1]] = morphpos
        #output[tasks[2]] = morphgender
        #output[tasks[3]] = morphnumber
        #output[tasks[4]] = morphperson
        #output[tasks[5]] = morphcase
        #output[tasks[6]] = morphvib
        output[tasks[1]] = chunk_

    return output


#text = 'इसे लेकर भास्कर ने पड़ताल की तो पता चला, देश में 16 ऐसी कंपनियां हैं जो वैक्सीन तुरंत बनाना शुरू कर सकती हैं।'
#print (shallow_parse(text))
#exit()


domain = sys.argv[2]

d_test = {}
count = 0
for test in tests:
    test = "/nmtltrc/vandan/tools/shallow_parser/"+domain+"/"+test.replace('news',domain)
    d_test[tasks[count]]=read_files(test)
    count = count + 1


out_files = {}
for task in tasks:
    _f = open('/nmtltrc/vandan/tools/shallow_parser/'+domain+'/'+task+'task_test.txt','w')
    out_files[task] =_f

inp = []
count = 0
for line in codecs.open(sys.argv[1]):
    line = line.strip().split('\t')
    if line==['']:
        count = count + 1
        sent = " ".join(inp)
        #if ':' in sent:
        print (count,':',sent.split())
        out = shallow_parse(sent)
        
        for o in out:
            for w in out[o]:
                out_files[o].write(w.replace('@#@','\t')+'\n')
            out_files[o].write('\n')
        inp = []
    else:
        if '.' in line[0] and '.'!=line[0]:
            line[0] = line[0].replace('.','')
        if ',' in line[0] and ','!=line[0]:
            line[0] = line[0].replace(',','')
        if '-' in line[0] and '-'!=line[0]:
            line[0] = line[0].replace('-','')
        if '_' in line[0] and '_'!=line[0]:
            line[0] = line[0].replace('_','')
        if ':' in line[0] and ':'!=line[0]:
            line[0] = line[0].replace(':','')
        
        if line[0].strip():
            inp.append(line[0].strip())

    #print (word)

    #print (out)
