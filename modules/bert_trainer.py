from logging import Logger
from tqdm.auto import tqdm
import math

import torch
from torch.utils.data import DataLoader
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, LabelAccuracyEvaluator
from sentence_transformers.readers import *

from modules.snli_utils import _read_tsv, _create_examples_snli, _create_examples_mnli


class BertTrainer:
    """
    Class to train Bert model
    :param logger: logger to use in model
    # TODO Finish it
    """
    def __init__(self, logger: Logger, train_path: str, dev_path: str, test_path: str, base_model: str,
                 batch_size: int, path_to_save: str, **kwargs):
        self.logger = logger
        self.logger.info("Models are loaded and ready to use.")

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path

        self.base_model = base_model
        self.batch_size = batch_size

        self.label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}

        self.path_to_save = path_to_save

    def initialize_model(self):
        # Read the dataset
        # Use BERT for mapping tokens to embeddings
        word_embedding_model = models.BERT(self.base_model)

        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        self.train_loss_nli = losses.SoftmaxLoss(model=self.model,
                                                 sentence_embedding_dimension=self.model.get_sentence_embedding_dimension(),
                                                 num_labels=len(self.label2int))

    def preparing_data(self):
        train_snli = _create_examples_snli(_read_tsv(self.train_path), 'train_s')
        dev_snli = _create_examples_snli(_read_tsv(self.dev_path), 'dev_s')
        test_snli = _create_examples_snli(_read_tsv(self.test_path), 'test_s')

        # Convert the dataset to a DataLoader ready for training
        self.logger.info("Read train dataset")

        train_nli_samples = []
        dev_nli_samples = []
        test_nli_samples = []

        for row in tqdm(train_snli):
            label_id = self.label2int[row[3]]
            train_nli_samples.append(InputExample(guid=row[0], texts=[row[1], row[2]], label=label_id))
        for row in tqdm(dev_snli):
            label_id = self.label2int[row[3]]
            dev_nli_samples.append(InputExample(guid=row[0], texts=[row[1], row[2]], label=label_id))
        for row in tqdm(test_snli):
            label_id = self.label2int[row[3]]
            test_nli_samples.append(InputExample(guid=row[0], texts=[row[1], row[2]], label=label_id))

        train_data_nli = SentencesDataset(train_nli_samples, model=self.model)
        self.train_dataloader_nli = DataLoader(train_data_nli, shuffle=True, batch_size=self.batch_size)
        dev_data_nli = SentencesDataset(dev_nli_samples, model=self.model)
        self.dev_dataloader_nli = DataLoader(dev_data_nli, shuffle=True, batch_size=self.batch_size)
        test_data_nli = SentencesDataset(test_nli_samples, model=self.model)
        self.test_dataloader_nli = DataLoader(test_data_nli, shuffle=True, batch_size=self.batch_size)

    def save_model(self):
        torch.save(self.train_loss_nli.classifier.cpu(), self.path_to_save + 'classifier_model')
        self.model.save(self.path_to_save + "bert_model_trained")

    def load_model(self, text_model_path, classifier_path):
        self.model = SentenceTransformer(text_model_path)
        self.classification_model = torch.load(classifier_path)
        self.train_loss_nli = losses.SoftmaxLoss(model=self.model,
                                                 sentence_embedding_dimension=self.model.get_sentence_embedding_dimension(),
                                                 num_labels=len(self.label2int))
        self.train_loss_nli.classifier = self.classification_model

    def train_model(self, number_of_epochs=1):
        warmup_steps = math.ceil(len(self.train_data_nli) * number_of_epochs / self.batch_size * 0.1)
        self.logger.info("Warmup-steps: {}".format(warmup_steps))
        train_objectives = [(self.train_dataloader_nli, self.train_loss_nli)]

        validation_performance = []
        test_performance = []

        test_evaluator = LabelAccuracyEvaluator(self.test_dataloader_nli,
                                                name='nli_test',
                                                softmax_model=self.train_loss_nli)
        dev_evaluator = LabelAccuracyEvaluator(self.dev_dataloader_nli,
                                               name='nli_test',
                                               softmax_model=self.train_loss_nli)

        validation_performance.append(self.model.evaluate(dev_evaluator))
        test_performance.append(self.model.evaluate(test_evaluator))
        print(f'Iteration - {0} ...')
        print(f'Validation performance - {validation_performance[-1]} ...')
        print(f'Test performance - {test_performance[-1]} ...')

        for i in range(1):
            self.model.fit(train_objectives=train_objectives)
            validation_performance.append(self.model.evaluate(dev_evaluator))
            test_performance.append(self.model.evaluate(test_evaluator))
            print(f'Iteration - {i+1} ...')
            print(f'Validation performance - {validation_performance[-1]} ...')
            print(f'Test performance - {test_performance[-1]} ...')