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


class WikiFactChecker(BaseModel):
    """
    Class exploit wikipedia api and BERT and verify the claim
    :param logger: logger to use in model
    :param bert_model_path: path to saved fine-tuned bert model
    :param classification_model_path: path to saved fine-tuned classification_model
    """
    def __init__(self, logger: Logger, bert_model_path: str, classification_model_path: str, **kwargs):
        super().__init__(logger, **kwargs)
        self.bert_model = SentenceTransformer(bert_model_path)

        if torch.cuda.is_available():
            self.classification_model = torch.load(classification_model_path).cuda()
        else:
            self.classification_model = torch.load(classification_model_path)

        self.logger.info("Models are loaded and ready to use.")

        self.candidate_picker = WikiCandidatesSelector(logger=self.logger)
        self.logger.info("Wiki candidates picker loaded")

    def predict_all(self, claim):
        classification_results = {}
        label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
        int2label = {v: k for k, v in label2int.items()}

        hypothesis = self.candidate_picker.get_candidates(claim)

        embedding_claim = self.bert_model.encode(claim, show_progress_bar=False, convert_to_numpy=False)

        for article, sentences in tqdm(hypothesis.items()):
            embeddings_hypothesis = self.bert_model.encode(sentences, show_progress_bar=True, convert_to_numpy=False)
            results = []
            for embedding in embeddings_hypothesis:
                stacked_features = self._vector_stacking_logic((embedding_claim, embedding))
                probs = self.classification_model(stacked_features).detach().cpu().numpy()
                final_label = int2label.get(np.argmax(probs))
                probs = self._normalize_probs(probs)
                results.append({"label": final_label,
                                'contradiction_prob': probs[0],
                                'entailment_prob': probs[1],
                                'neutral_prob': probs[2]})
            classification_results[article] = results

        return self.convert_to_one_res(hypothesis, classification_results)

    @staticmethod
    def convert_to_one_res(res1, res2):
        all_dfs = []
        for k in res1.keys():
            df1 = pd.DataFrame(res1[k])
            df1.columns = ['text']
            df2 = pd.DataFrame(res2[k])
            df1['article'] = k
            dfFull = pd.concat([df1, df2], axis=1)
            all_dfs.append(dfFull)
        return pd.concat(all_dfs, axis=0).to_dict('records')

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
