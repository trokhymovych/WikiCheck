import torch
import numpy as np
from typing import *
from logging import Logger

from sentence_transformers import SentenceTransformer

from modules.utils.logging_utils import DEFAULT_LOGGER, DEFAULT_MEASURER


class SentenceNLIModel:
    """
    Class to use transformer sentence based NLI model on inference
    :param logger: logger to use in model
    :param bert_model_path: path to saved fine-tuned bert model
    :param classification_model_path: path to saved fine-tuned classification_model
    """
    def __init__(self, bert_model_path: str, classification_model_path: str,
                 logger: Logger = DEFAULT_LOGGER, label2int: dict = None, **kwargs):

        self.logger = logger
        self.profiler = kwargs.get('profiler', DEFAULT_MEASURER)

        self.bert_model = SentenceTransformer(bert_model_path)
        if torch.cuda.is_available():
            self.classification_model = torch.load(classification_model_path).cuda()
        else:
            self.classification_model = torch.load(classification_model_path)
        self.logger.info("Models are loaded and ready to use.")

        if not label2int:
            self.label2int = {"SUPPORTS": 1, "REFUTES": 0, "NOT ENOUGH INFO": 2}
        else:
            self.label2int = label2int
        self.int2label = {v: k for k, v in self.label2int.items()}
        self.logger.info(f"Labels encoding is {self.label2int}")

    def predict(self, text="", hypothesis=""):

        embeddings = self.bert_model.encode([text, hypothesis], show_progress_bar=False, convert_to_numpy=False)
        stacked_features = self._vector_stacking_logic(embeddings)

        probs = self.classification_model(stacked_features).detach().cpu().numpy()
        final_label = self.int2label.get(np.argmax(probs))
        probs = self._normalize_probs(probs)

        return {"label": final_label,
                'contradiction_prob': probs[0],
                'entailment_prob': probs[1],
                'neutral_prob': probs[2]}

    def predict_batch(self, texts, hypothesis):

        if isinstance(hypothesis, str):
            raise AttributeError("Hypothesis should be a list in case of batch processing.")

        self.profiler.start_measure_local('embedding_claim')
        embeddings_claim = self.bert_model.encode(texts, show_progress_bar=False, convert_to_numpy=False)
        self.profiler.finish_measure_local()

        if isinstance(texts, str):
            embeddings_texts = [embeddings_claim] * len(hypothesis)
        else:
            embeddings_texts = embeddings_claim

        self.profiler.start_measure_local('embedding_hypothesis')
        embeddings_hypothesis = self.bert_model.encode(hypothesis, show_progress_bar=False, convert_to_numpy=False)
        self.profiler.finish_measure_local()

        results = []
        self.profiler.start_measure_local('classification')
        for embeddings in zip(embeddings_texts, embeddings_hypothesis):
            stacked_features = self._vector_stacking_logic(embeddings)
            results.append(self._normalize_probs(self.classification_model(stacked_features).detach().cpu().numpy()))
        self.profiler.finish_measure_local()
        return results

    @staticmethod
    def _vector_stacking_logic(vectors) -> np.array:
        """
        Implements vectors stacking logic after bert model before classifier
        :param vectors: list of two embedding corresponds to two input vectors.
        """
        return torch.cat([vectors[0], vectors[1], torch.abs(vectors[0] - vectors[1])], dim=0)

    @staticmethod
    def _normalize_probs(data):
        """
        Method to normalize model output in probability-like format (sum is 1 and every element is in [0,1])
        :param data: list of numbers
        """
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return data / data.sum()
