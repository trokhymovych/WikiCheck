import datetime
import json
from argparse import ArgumentParser
import sys

from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_restx import Api, Resource, fields, marshal
from flask_cors import CORS
from pathlib import Path

# for debug mode
# sys.path.insert(0, "/Users/ntr/Documents/tresh/fairapi")
sys.path.insert(0, "/home/trokhymovych/fairapi")
from modules.sentence_bert_model import SentenceBertModel
from modules.complex_model import WikiFactChecker
from modules.logging_utils import get_logger, check_if_none, ROOT_LOGGER_NAME

parser = ArgumentParser()
parser.add_argument('--config', type=str, required=False,
                    default='modules/configs/sentence_bert_config.json', help='path to config')

args = parser.parse_args()
config_path = args.config
logger = get_logger(name=ROOT_LOGGER_NAME,
                    console=True,
                    log_level="INFO",
                    propagate=False)

logger.info(f"Reading config from {Path(config_path).absolute()}")
with open(config_path) as con_file:
    config = json.load(con_file)
logger.info(f"Using config {config}")

logger.info(f"Loading model {config.get('model_name')}...")
model = SentenceBertModel(logger, **config)
complex_model = WikiFactChecker(logger, **config)

# setting the api
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
CORS(app)
api = Api(app, version=config.get("api_version", "0.0"), title='Wikipedia fact checking API')
ns1 = api.namespace('nli_model', description=config.get('model_name', 'Wikipedia NLI model'))
ns2 = api.namespace('fact_checking_model', description='Wikipedia Fact checking model')

# response format
response = api.model('model_response', {
    'label': fields.String(required=True, description='classification label'),
    'contradiction_prob': fields.Float(required=True, description='contradiction class probability'),
    'entailment_prob': fields.Float(required=True, description='entailment class probability'),
    'neutral_prob': fields.Float(required=True, description='neutral class probability'),
})

response_full = api.model('Record', {
    "text": fields.String(required=True, description=''),
    "article": fields.String(required=True, description=''),
    "label": fields.String(required=False, description=''),
    "contradiction_prob": fields.Float(required=True, description=''),
    "entailment_prob": fields.Float(required=True, description=''),
    "neutral_prob": fields.Float(required=True, description=''),
})

response_model = api.model("Result", {
    'results': fields.List(fields.Nested(response_full))
})


@ns1.route('/')
class TodoList(Resource):

    @ns1.doc('trigger_model')
    @ns1.param('text', _in='query')
    @ns1.param('hypothesis', _in='query')
    @ns1.marshal_list_with(response)
    def get(self):
        start_time = datetime.datetime.now()
        text = request.args.get('text')
        hypothesis = request.args.get('hypothesis')

        text = check_if_none(text)
        hypothesis = check_if_none(hypothesis)

        logger.info(f'Query with params={{text: {text}, hypothesis: {hypothesis}}}')
        result = model.predict(text, hypothesis)

        end_time = datetime.datetime.now()
        dif_time = str(end_time - start_time)

        logger.info(f'API; ModelOne Get response; difference: {dif_time}')

        return result


@ns2.route('/')
class TodoList(Resource):

    @ns2.doc('trigger_model')
    @ns2.param('claim', _in='query')
    @ns2.marshal_with(response_model)
    def get(self):
        start_time = datetime.datetime.now()
        claim = request.args.get('claim')

        claim = check_if_none(claim)

        logger.info(f'Query with params={{text: {claim}}}')
        result = complex_model.predict_all(claim)

        end_time = datetime.datetime.now()
        dif_time = str(end_time - start_time)

        logger.info(f'API; ModelFull Get response; difference: {dif_time}')

        logger.info(f'API; ModelFull sending the response')
        # The kwarg envelope does the trick
        return {'results': result}

    def put(self):
        print(complex_model.profiler.times_global)
        complex_model.dump_time_stats()


if __name__ == '__main__':
    app.run(debug=True, port=80, host="0.0.0.0", threaded=True)
