import numpy as np
from logging import Logger
from tqdm.auto import tqdm

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

        self.profiler = TimeMeasurer(save_path='')
        self.logger.info(f"Time logger is loaded, with logging mode {time_measurer_config.get('mode_on', False)}")

        self.model_level_one = WikiCandidatesSelector(**model_level_one_config)
        self.logger.info("Model level one is loaded.")
        self.model_level_two = SentenceNLIModel(**model_level_two_config)
        self.logger.info("Model level two is loaded.")

    def predict_all(self, claim):

        self.profiler.begin_sample(claim)

        hypothesis = self.model_level_one.get_candidates(claim)

        all_sentences = []
        all_articles = []
        for article, sentences in tqdm(hypothesis.items()):
            all_sentences += sentences
            all_articles += [article] * len(sentences)

        probabilities = self.model_level_two.predict_batch(claim, all_sentences)

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

    def dump_time_stats(self):
        self.profiler.finish_logging_time()
