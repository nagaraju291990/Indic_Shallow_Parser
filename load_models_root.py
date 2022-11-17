import tensorflow_text as tftxt
import tensorflow as tf
import numpy as np


vocab_file = "models/spm.model"
_tokenizer = tftxt.SentencepieceTokenizer(
    model=tf.io.gfile.GFile(vocab_file, 'rb').read(), add_eos=True)

empty_str_tokenized = _tokenizer.tokenize("").numpy()
_eos_id = empty_str_tokenized.item()

def _decode(ids):
    return _tokenizer.detokenize(ids).numpy().decode()

def _trim_and_decode(ids):
    ids = list(ids)
    return _decode(ids)

model_path = 'models/hi_morph_word2root/'
model = tf.saved_model.load(model_path)
his = {}

def get_root(_input):
    send = []
    for text in _input.split():
        if text in his:
            send.append(his[text])
        else:
            ids = _tokenizer.tokenize(text)
            ids = tf.constant(np.array([ids]))
            serving_fn = model.signatures["serving_default"]
            all_outputs = serving_fn(tf.dtypes.cast(ids, dtype=tf.int64))
            all_outputs = all_outputs['outputs']
            translation = _trim_and_decode(all_outputs.numpy()[0])
            if ' ' in translation.strip():
                send.append(text+':'+text)
                his[text]=text+':'+text
            else:
                send.append(text+':'+translation)
                his[text]=text+':'+translation
    return send

def get_root_batch(_input):
    send = []
    ids = [tf.constant(np.array(_tokenizer.tokenize(w))) for w in _input.split()]
    print (ids)
    serving_fn = model.signatures["serving_default"]
    all_outputs = serving_fn(tf.dtypes.cast(ids, dtype=tf.int64))
    all_outputs = all_outputs['outputs']
    print (all_outputs.numpy())
    translation = _trim_and_decode(all_outputs.numpy())
    #print (translation[0].decode())
    send.append(text+':'+translation)
    return send



text = "ट्रेन में होने वाले हर अपराध में इस गाँव का कोई न कोई शामिल मिल जाएगा ।"
#text = "जम्मू में कांग्रेस नेताओं ने पार्टी की कार्य प्रणाली पर गंभीर सवाल उठाए हैं . "
text = "नेताओं"
#print (text)
#print (get_root_batch(text))



