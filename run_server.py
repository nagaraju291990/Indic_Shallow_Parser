
import os
import time
from typing import Dict
from flask import request, jsonify, Flask
#from load_models import shallow_parse
#from load_models_all import shallow_parse
#from load_models_all_v1 import shallow_parse
#from load_models_ilci_v3 import shallow_parse
#from load_models_ilcicourse_v3 import shallow_parse
from load_models_langidentify import language_identify
from load_models_all_v3 import shallow_parse

class LocalFlask(Flask):

    def __init__(self, name):
        super().__init__(name)
        self.add_routes()

    def add_basic_routes(self):
        self.add_url_rule('/', 'index', self.index)
        self.add_url_rule('/health', 'health', self.health)
        self.add_url_rule('/readiness', 'readiness', self.readiness_check)
        self.add_url_rule('/liveness', 'liveness', self.liveness_check)
        self.add_url_rule('/status', 'status', self.status)

    def index(self):
        return 'Welcome to oneMT Server : ' + self.name

    @staticmethod
    def health():
        return jsonify(['ok'])

    # pylint: disable=no-self-use
    def liveness_check(self):
        return jsonify(['ok'])

    def readiness_check(self):
        if self.is_ready():
            return '', 204

        return jsonify(self.get_status()), 503

    def status(self):
        return jsonify(self.get_status())

    # pylint: disable=no-self-use
    def get_status(self) -> Dict[str, str]:
        return {}

    def is_ready(self) -> bool:
        return True

    def add_routes(self):
        self.add_basic_routes()
        self.add_url_rule('/parser', 'parser', self.parser, methods=["POST"])
        self.add_url_rule('/langidentify', 'langidentify', self.langidentify, methods=["POST"])

    def error_models(self) -> Dict[str, str]:
        return {'error': 'models are not present'}

    def error(self, error_message) -> Dict[str, str]:
        return {'error': error_message}

    def parser(self):
        start_time = time.time()
        request_json = request.json or {}
        text = request_json.get('text')
        tokens = request_json.get('tokens') or []
        mode = request_json.get('mode')
        language = request_json.get('language')
        model_id = request_json.get('model_id')

        if tokens:
            text = " ".join(tokens)
        if not mode:
            mode ='SSF'
        print("incoming request : %s", request_json)
        if not text or not language:
            print('text or language not specified: %s', request_json)
            return self.error('text or language not specified , mode can be _____'), 400

        #response = shallow_parse(text, language, mode, model_id)
        response = shallow_parse(text, language)
        print (response)
        #print("Response:: %s", response.to_dict())
        #return jsonify(response), 400 if response.has_error() else 200
        return jsonify(response), 200

    def langidentify(self):
        start_time = time.time()
        request_json = request.json or {}
        text = request_json.get('text') or request_json.get('input')
        out = {}
        print("incoming request : %s", request_json)
        if not text:
            print('text not specified: %s', request_json)
            return self.error('text not specified , mode can be _____'), 400
        out["output"]=[]
        if type(text)==list:
            for inst in text:
                response = language_identify(inst["source"])
                out['output'].append({
                        "source":inst["source"],
                        "langPrediction":[{"langCode" : response, "langScore" : 100 }]
                })
        print("Response:: %s", out)
        return jsonify(out), 200


app = LocalFlask(__name__)
print('** Initialization complete.')
