import numpy as np
import pandas as pd
from logging import Logger
from tqdm.auto import tqdm
import pickle

from modules.model_level_one import WikiCandidatesSelector
from modules.model_level_two import SentenceNLIModel
from modules.utils.measurer import TimeMeasurer
from modules.utils.logging_utils import DEFAULT_LOGGER, DEFAULT_MEASURER


class WikiFactChecker:
    """
    Class exploit wikipedia api and BERT and verify the claim
    :param logger: logger to use in model
    :param bert_model_path: path to saved fine-tuned bert model
    :param classification_model_path: path to saved fine-tuned classification_model
    """
    def __init__(self, config, logger: Logger = DEFAULT_LOGGER, **kwargs):

        self.logger = logger
        time_measurer_config = config.get('measurer', dict())
        model_level_one_config = config.get('model_level_one', dict())
        model_level_two_config = config.get('model_level_two', dict())
        aggregation_model_path = config.get('aggregation', None)

        self.profiler = TimeMeasurer(save_path='')
        self.logger.info(f"Time logger is loaded, with logging mode {time_measurer_config.get('mode_on', False)}")

        self.model_level_one = WikiCandidatesSelector(**model_level_one_config)
        self.logger.info("Model level one is loaded.")
        self.model_level_two = SentenceNLIModel(**model_level_two_config)
        self.logger.info("Model level two is loaded.")

        if aggregation_model_path:
            with open(aggregation_model_path, 'rb') as handle:
                models_dict = pickle.load(handle)
                self.model_clf = models_dict['clf_model']
                self.model_ranking = models_dict['ranking_model']
            self.logger.info("Aggregation stage models are loaded")
        else:
            self.logger.info("Aggregation stage is not loaded")

    def predict_all(self, claim):

        self.profiler.begin_sample(claim)

        hypothesis = self.model_level_one.get_candidates(claim)

        all_sentences = []
        all_articles = []
        for article, sentences in tqdm(hypothesis.items()):
            all_sentences += sentences
            all_articles += [article] * len(sentences)

        probabilities = self.model_level_two.predict_batch(claim, all_sentences)['probs']

        results = []
        sorting_logic = []
        for p, text, article in zip(probabilities, all_sentences, all_articles):
            final_label = self.model_level_two.int2label.get(np.argmax(p))
            results.append({"claim": claim,
                            "text": text,
                            "article": article,
                            "label": final_label,
                            'contradiction_prob': p[0],
                            'entailment_prob': p[1],
                            'neutral_prob': p[2]})

            sorting_logic.append(np.max([p[1], p[0]]))
        ids = np.argsort(sorting_logic)
        sorted_results = [results[i] for i in ids[::-1]]
        self.profiler.end_sample()
        return sorted_results

    def predict_and_aggregate(self, claim):
        hypothesis = self.model_level_one.get_candidates(claim)

        all_sentences = []
        all_articles = []
        for article, sentences in tqdm(hypothesis.items()):
            all_sentences += sentences
            all_articles += [article] * len(sentences)

        model_two_res = self.model_level_two.predict_batch(claim, all_sentences, return_cosine=True)
        probabilities, cosines = model_two_res['probs'], model_two_res['cosines']

        results = []
        sorting_logic = []
        for p, cos, text, article in zip(probabilities, cosines, all_sentences, all_articles):
            final_label = self.model_level_two.int2label.get(np.argmax(p))
            results.append({"claim": claim,
                            "text": text,
                            "article": article,
                            "label": final_label,
                            'contradiction_prob': p[0],
                            'entailment_prob': p[1],
                            'neutral_prob': p[2],
                            'cos': cos})

            sorting_logic.append(np.max([p[1], p[0]]))
        ids = np.argsort(sorting_logic)
        sorted_results = [results[i] for i in ids[::-1]]

        return self.strategy_catboost(sorted_results)

    def strategy_catboost(self, res):
        k = 10
        led = {'SUPPORTS': 1, 'REFUTES': 0}
        try:
            a = pd.DataFrame(res).sort_values('cos', ascending=False)
            if len(a) < k:
                a = pd.concat([a, pd.DataFrame(np.zeros((k - len(a), len(a.columns))), columns=a.columns)])
            features_lable = list(a.head(k).cos) + list(a.head(k).contradiction_prob) + list(
                a.head(k).entailment_prob) + list(a.head(k).neutral_prob)
            lable = self.model_clf.predict(features_lable)[0]

            # evidence prediction
            if lable == 'NOT ENOUGH INFO':
                evidences = []
            else:
                features_lable = []
                for i, row in a.iterrows():
                    features_lable.append(
                        [row.cos, row.contradiction_prob, row.entailment_prob, row.neutral_prob, led[lable]])

                score_preds = np.array(self.model_ranking.predict(features_lable))
                evidence_ids = score_preds.argsort()[-5:][::-1]
                evidence_df = a.iloc[evidence_ids]
                evidences = [[str(article), text] for article, text in
                             zip(evidence_df.article.values, evidence_df['text'].values)]

            return {"predicted_label": lable, "predicted_evidence": evidences}
        except Exception as e:
            self.logger.error(e)
            return {"predicted_label": "NOT ENOUGH INFO", "predicted_evidence": []}

    def dump_time_stats(self):
        self.profiler.finish_logging_time()
