import torch
import numpy as np
from typing import *
from logging import Logger
import pickle

from sentence_transformers import SentenceTransformer

from modules.base_model import BaseModel


class SentenceBertModel(BaseModel):
    """
    Class to use bert sentence based model on inference
    :param logger: logger to use in model
    :param bert_model_path: path to saved fine-tuned bert model
    :param classification_model_path: path to saved fine-tuned classification_model
    """
    def __init__(self, logger: Logger, bert_model_path: str, classification_model_path: str, cache_path: str, **kwargs):
        super().__init__(logger, **kwargs)
        self.bert_model = SentenceTransformer(bert_model_path)
        self.classification_model = torch.load(classification_model_path)
        self.logger.info("Models are loaded and ready to use.")

        self.logger.info("Loading cache...")
        if cache_path == "":
            self.logger.info("Cache is not used.")
        else:
            with open(cache_path, 'rb') as handle:
                self.cache = pickle.load(handle)
            self.logger.info("Cache loaded...")

    def predict(self, text="", hypothesis=""):
        label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
        int2label = {v: k for k, v in label2int.items()}

        embeddings = self._encode_texts([text, hypothesis])
        stacked_features = self._vector_stacking_logic(embeddings)

        probs = self.classification_model(stacked_features).detach().numpy()
        final_label = int2label.get(np.argmax(probs))
        probs = self._normalize_probs(probs)

        return {"label": final_label,
                'contradiction_prob': probs[0],
                'entailment_prob': probs[1],
                'neutral_prob': probs[2]}

    def _encode_texts(self, texts: List[str]) -> List[np.array]:
        """
        Make embeddings for list of texts that correspond to text and hypothesis
        :param texts: list of two strings
        """
        results = []
        for t in texts:
            results.append(self.cache.get(t, self.bert_model.encode(t,
                                                                    show_progress_bar=False,
                                                                    convert_to_numpy=False)))
        return results

    @staticmethod
    def _vector_stacking_logic(vectors: List[np.array]) -> np.array:
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
