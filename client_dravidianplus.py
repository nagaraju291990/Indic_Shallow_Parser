import sys
import codecs
import requests
import json
import ast
import glob

files = glob.glob(sys.argv[1]+"/*.txt")


url = "http://0.0.0.0:8046/parser"
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}

mt_hist = {}

def call_parser(text, language='eng'):
    data = {'text': text, 'language':language, 'mode':'list'}
    r = requests.post(url, headers=headers, json=data)
    output = json.loads(r.text)['text']
    return output

sent = []
asitis = []

for _file in files:
    print (_file)
    if '_train-' in _file or '_out.' in _file:
        continue

    out_file = _file+'_out.multidravidianplusmodel_muril.txt'
    ff = open(out_file,'w')
    language = _file.split('/')[-1].split('_')[0]
    for line in codecs.open(_file):
        line = line.strip()
        if line=='':
            text = " ".join(sent).strip()
            #print (text)
            if text:
                output = call_parser(text, language='dravidianplus')
                output = ast.literal_eval(output)
                output = output[0]['pos']
                for itm,pred in zip(asitis,output):
                    a, b = itm 
                    c, d = pred.split('G:$:$:G')
                    print (a+'\t'+b+'\t'+d+'\n')
                    ff.write(a+'\t'+b+'\t'+d+'\n')
                print ()
                ff.write('\n')
            sent = []
            asitis = []
        else:
            sent.append(line.split('\t')[0])
            asitis.append(line.split('\t'))

    ff.close()
