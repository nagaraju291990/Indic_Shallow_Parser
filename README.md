Indic Shallow Parser for Indian Languages

This repository provides a production ready shallow parsing toolkit for Indian languages with a focus on Hindi. The system performs part of speech tagging chunking morphological analysis and root extraction using multi task learning based neural models. The toolkit supports multiple output formats and both sentence level and file based processing.

The work is based on peer reviewed research published in ACM Transactions on Asian and Low Resource Language Information Processing.

---

Overview

Shallow parsing is a core natural language processing task consisting of part of speech tagging and chunking. This implementation extends shallow parsing by including detailed morphological analysis and root extraction. The models are trained using multi task learning and contextual embeddings and are designed for realistic sentence level evaluation.

The toolkit currently supports Hindi and is designed to be extensible to other Indian languages.

---

Features

Hindi part of speech tagging  
Chunking  
Morphological analysis including gender number person case and vibhakti  
Root or lemma extraction using a neural model  
Multiple output formats JSON SSF and CoNLL  
Line by line file processing  
Reusable model and token caching  
Optional GPU support using TensorFlow  

---

Repository Structure

Indic_Shallow_Parser  
shallow_parser core python package  
indic_nlp_resources indic nlp data  
labels pos chunk and morphology labels  
models downloaded tensorflow models  
download_models.sh model download script  
run_conll.py file to conll conversion  
test_run.py example usage  
requirements.txt python dependencies  

---

Installation

Create a virtual environment

python3 -m venv venv  
source venv/bin/activate  

Install dependencies

pip install -r requirements.txt  

For GPU support CUDA 11 and cuDNN 8 are required. CPU only usage works without any additional setup.

---

Download Models

Run the following command from the project root

chmod +x download_models.sh  
./download_models.sh  

This downloads and places all required models in the correct directory structure expected by the code.

---

Usage

Python API usage

from shallow_parser import shallow_parse  

text = "ट्रेन में होने वाले हर अपराध में इस गाँव का कोई न कोई शामिल मिल जाएगा ।"  

JSON output  
result = shallow_parse(text "hin" mode="json")  

SSF output  
ssf = shallow_parse(text "hin" mode="ssf")  
print("\n".join(ssf))  

CoNLL output  
conll = shallow_parse(text "hin" mode="conll")  
print("\n".join(conll))  

---

File Based Processing

Input file example input.txt

ट्रेन में होने वाले हर अपराध में इस गाँव का कोई न कोई शामिल मिल जाएगा ।  
भारत एक लोकतांत्रिक देश है ।  

Run

python run_conll.py  

Output is written to output.conll with one sentence per block and a blank line between sentences following standard CoNLL conventions.

---

Output Formats

JSON  
Structured dictionary containing token wise pos chunk morphology and root information  

SSF  
Shakti Standard Format compatible with Indian language treebanks  

CoNLL  
Tab separated format suitable for corpus creation model training and linguistic analysis  

---

Models

Hindi shallow parser trained using multi task learning  
Hindi morphological root generation model  
SentencePiece tokenizer  

Models are downloaded automatically using the provided script.

---

Citation

If you use this toolkit or models in your research please cite the following paper

Mishra Pruthwik Mujadia Vandan Sharma Dipti Misra  
Multi Task Learning Based Shallow Parsing for Indian Languages  
ACM Transactions on Asian and Low Resource Language Information Processing  
Volume 23 Number 9 Article 131 August 2024  
DOI 10.1145/3664620  

BibTeX

@article{10.1145/3664620,
author = {Mishra, Pruthwik and Mujadia, Vandan and Sharma, Dipti Misra},
title = {Multi Task Learning Based Shallow Parsing for Indian Languages},
year = {2024},
journal = {ACM Trans. Asian Low-Resour. Lang. Inf. Process.},
volume = {23},
number = {9},
articleno = {131},
doi = {10.1145/3664620}
}

---

Maintainer

Maintainer  
Vandan Mujadia  

For any questions issues or suggestions please open a GitHub issue in this repository.  
Research and collaboration related queries can also be addressed to the maintainer via the associated academic channels.

---

License

This repository is intended for research and academic use.  
Model and data licenses follow their respective original sources.

