import torch
import numpy as np
from typing import *
from logging import Logger
from tqdm.auto import tqdm
import pickle
import pandas as pd

from sentence_transformers import SentenceTransformer

from modules.base_model import BaseModel
from modules.candidatets_picker import WikiCandidatesSelector
from modules.measurer import TimeMeasurer


class WikiFactChecker(BaseModel):
    """
    Class exploit wikipedia api and BERT and verify the claim
    :param logger: logger to use in model
    :param bert_model_path: path to saved fine-tuned bert model
    :param classification_model_path: path to saved fine-tuned classification_model
    """
    def __init__(self, logger: Logger, bert_model_path: str, classification_model_path: str, **kwargs):
        super().__init__(logger, **kwargs)

        self.label2int = {"SUPPORTS": 1, "REFUTES": 0, "NOT ENOUGH INFO": 2}

        self.profiler = TimeMeasurer(save_path='')
        self.logger.info("Time logger is loaded")

        self.bert_model = SentenceTransformer(bert_model_path)
        if torch.cuda.is_available():
            self.classification_model = torch.load(classification_model_path).cuda()
        else:
            self.classification_model = torch.load(classification_model_path)
        self.logger.info("NLI Models are loaded and ready to use.")

        self.candidate_picker = WikiCandidatesSelector(**{'logger': self.logger, 'profiler': self.profiler})
        self.logger.info("Wiki candidates picker loaded")

    def predict_all(self, claim):
        self.profiler.begin_sample(claim)
        classification_results = {}
        int2label = {v: k for k, v in self.label2int.items()}

        hypothesis = self.candidate_picker.get_candidates(claim)

        self.profiler.start_measure_local('embedding_claim')
        embedding_claim = self.bert_model.encode(claim, show_progress_bar=False, convert_to_numpy=False)
        self.profiler.finish_measure_local()

        self.profiler.start_measure_local('embedding_hypothesis')
        all_sentences = []
        all_articles = []
        for article, sentences in tqdm(hypothesis.items()):
            all_sentences += sentences
            all_articles += [article] * len(sentences)
        embeddings_hypothesis = self.bert_model.encode(all_sentences, show_progress_bar=True, convert_to_numpy=False)
        self.profiler.finish_measure_local()

        self.profiler.start_measure_local('classification')
        results = []
        for embedding, text, article in zip(embeddings_hypothesis, all_sentences, all_articles):
            stacked_features = self._vector_stacking_logic((embedding_claim, embedding))
            probs = self.classification_model(stacked_features).detach().cpu().numpy()
            final_label = int2label.get(np.argmax(probs))
            probs = self._normalize_probs(probs)
            results.append({"text": text,
                            "article": article,
                            "label": final_label,
                            'contradiction_prob': probs[0],
                            'entailment_prob': probs[1],
                            'neutral_prob': probs[2]})
        self.profiler.finish_measure_local()

        self.profiler.end_sample()
        return results

    def dump_time_stats(self,):
        self.profiler.finish_logging_time()

    @staticmethod
    def _vector_stacking_logic(vectors) -> np.array:
        """
        Implements vectors stacking logic after bert model before classifier
        :param vectors: list of two embedding corresponds to two input vectors.
        """
        return torch.cat([vectors[0], vectors[1], np.abs(vectors[0] - vectors[1])], dim=0)

    @staticmethod
    def _normalize_probs(data):
        """
        Method to normalize model output in probability-like format (sum is 1 and every element is in [0,1])
        :param data: list of numbers
        """
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return data / data.sum()
