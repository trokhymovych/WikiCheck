import datetime
import json
from argparse import ArgumentParser

from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from pathlib import Path

from modules.model_complex import WikiFactChecker
from modules.model_level_two import SentenceNLIModel
from modules.utils.logging_utils import get_logger, check_if_none, ROOT_LOGGER_NAME, CSVLogger

parser = ArgumentParser()
parser.add_argument('--config', type=str, required=False,
                    default='configs/inference/sentence_bert_config.json', help='path to config')

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


logger.info(f"Loading models ...")
complex_model = WikiFactChecker(config, logger=logger)
file_logger = CSVLogger(config)
logger.info(f"Models loaded.")

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
CORS(app)

api = Api(app, version=config.get("api_version", "0.4"), title='WikiCheck API')

ns1 = api.namespace('nli_model', description=config.get('model_name', 'NLI model'))
ns2 = api.namespace('fact_checking_model', description='Fact checking model')
ns3 = api.namespace('fact_checking_aggregated', description='Fact checking model with aggregation')

response = api.model('model_response', {
    'label': fields.String(required=True, description='classification label'),
    'contradiction_prob': fields.Float(required=True, description='contradiction class probability'),
    'entailment_prob': fields.Float(required=True, description='entailment class probability'),
    'neutral_prob': fields.Float(required=True, description='neutral class probability'),
})

response_full = api.model('Record', {
    "claim": fields.String(required=True, description='Claim'),
    "text": fields.String(required=True, description='Hypothesis'),
    "article": fields.String(required=True, description='article name'),
    "label": fields.String(required=False, description='Predicted label'),
    "contradiction_prob": fields.Float(required=True, description=''),
    "entailment_prob": fields.Float(required=True, description=''),
    "neutral_prob": fields.Float(required=True, description=''),
})

response_model = api.model("Result", {
    'results': fields.List(fields.Nested(response_full))
})

response_aggregated = api.model("Aggregated_result", {
    "predicted_label": fields.String(required=True, description='Claim'),
    'predicted_evidence': fields.List(fields.List(fields.String()))
})

@ns1.route('/')
class TodoList(Resource):

    @ns1.doc('trigger_model')
    @ns1.param('claim', _in='query')
    @ns1.param('hypothesis', _in='query')
    @ns1.marshal_list_with(response)
    def get(self):
        start_time = datetime.datetime.now()
        claim = request.args.get('claim')
        hypothesis = request.args.get('hypothesis')

        text = check_if_none(claim)
        hypothesis = check_if_none(hypothesis)

        logger.info(f'Query with params={{text: {text}, hypothesis: {hypothesis}}}')
        result = complex_model.model_level_two.predict(text, hypothesis)

        end_time = datetime.datetime.now()
        dif_time = str(end_time - start_time)

        logger.info(f'[MODEL_LEVEL_TWO] API; ModelOne Get response; difference: {dif_time}')
        logger.info(f'[MODEL_LEVEL_TWO] API; ModelFull sending the response')

        params_to_log = {
            "datetime": str(datetime.datetime.now()),
            "model_name": "MODEL_LEVEL_TWO",
            "request": str({"text": text, "hypothesis": hypothesis}),
            "response": str(result),
            "time_spend": str(dif_time),
            "ip": str(request.remote_addr)
        }
        file_logger.add_log(params_to_log)

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

        logger.info(f'[COMPLEX MODEL] API; ModelFull Get response; difference: {dif_time}')
        logger.info(f'[COMPLEX MODEL] API; ModelFull sending the response')

        params_to_log = {
            "datetime": str(datetime.datetime.now()),
            "model_name": "COMPLEX_MODEL",
            "request": str({"claim": claim}),
            "response": str({'results': result[:10]}),
            "time_spend": str(dif_time),
            "ip": str(request.remote_addr)
        }
        file_logger.add_log(params_to_log)

        return {'results': result}


@ns3.route('/')
class TodoList(Resource):

    @ns3.doc('trigger_model')
    @ns3.param('claim', _in='query')
    @ns3.marshal_with(response_aggregated)
    def get(self):
        start_time = datetime.datetime.now()
        claim = request.args.get('claim')
        claim = check_if_none(claim)

        logger.info(f'Query with params={{text: {claim}}}')
        result = complex_model.predict_and_aggregate(claim)

        end_time = datetime.datetime.now()
        dif_time = str(end_time - start_time)

        logger.info(f'[COMPLEX MODEL. Aggregated] API; ModelFull Get response; difference: {dif_time}')
        logger.info(f'[COMPLEX MODEL. Aggregated] API; ModelFull sending the response')

        params_to_log = {
            "datetime": str(datetime.datetime.now()),
            "model_name": "COMPLEX_MODEL_AGGREGATED",
            "request": str({"claim": claim}),
            "response": str(result),
            "time_spend": str(dif_time),
            "ip": str(request.remote_addr)
        }
        file_logger.add_log(params_to_log)

        return result


if __name__ == '__main__':
    app.run(debug=False, port=80, host="0.0.0.0", threaded=True)

