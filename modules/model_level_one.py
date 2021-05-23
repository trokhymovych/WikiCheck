from typing import *

from flair.data import Sentence
from flair.models import SequenceTagger
from mediawiki import MediaWiki

from modules.utils.logging_utils import DEFAULT_LOGGER, DEFAULT_MEASURER


class WikiCandidatesSelector:
    """
    Class responsible for model of candidates selection from Wikipedia (Model level one)
    :param logger: logger to use in model
    :param separate: if make separate queries for each found entity
    :param n: number of results return after each query
    """

    def __init__(self, logger=DEFAULT_LOGGER, separate: bool = True, n: int = 3, **kwargs):

        self.profiler = kwargs.get('profiler', DEFAULT_MEASURER)
        self.logger = logger

        self.tagger = SequenceTagger.load('ner-fast')
        self.wikipedia = MediaWiki()
        self.separate = separate
        self.n = n

        self.logger.info("Candidate selector is loaded and ready to use.")

    def get_wiki_candidates_raw(self, query: str) -> List[str]:
        """
        Query Wikipedia with given text query
        :param query: text to query in Wikipedia
        :return list of links found for query
        """

        search_results = self.wikipedia.search(query, results=self.n)
        return [t.replace(' ', '_') for t in search_results]

    def get_entities(self, text: str) -> List[str]:
        """
        Get the list of named entities for given text.
        COMMENT: We should reinitialize this method for using another NER model
        :param text: str, text used for NER extraction
        :return list of str (entities found in text)
        """

        sentence = Sentence(text)
        self.tagger.predict(sentence)
        entities = []
        for entity in sentence.get_spans('ner'):
            entities.append(entity.text)
        return entities

    def get_wiki_candidates_NER(self, query: str) -> Set[str]:
        """
        Method to get the Wikipedia articles candidates with use of NER model
        :param query: str query claim
        :return set of links found for query
        """
        self.profiler.start_measure_local('NER_model')
        entities = self.get_entities(query)
        self.profiler.finish_measure_local()

        self.profiler.start_measure_local('wiki_search')
        search_results = self.get_wiki_candidates_raw(query)
        if not self.separate:
            search_results_en = self.get_wiki_candidates_raw(' '.join(entities))
        else:
            search_results_en = []
            for e in entities:
                search_results_en += self.get_wiki_candidates_raw(e)
        self.profiler.finish_measure_local()

        return set([t for t in search_results + search_results_en])

    def get_wiki_texts(self, articles_names: Set[str]) -> Dict:
        """
        Method that gets Wikipedia texts for given articles names if exist
        :param articles_names set of names for Wikipedia articles.
        :return the dict with article names as keys and list of related sentences as values
        """
        result = {}
        for name in articles_names:
            try:
                page = self.wikipedia.page(name)
                result[name] = page.summary.replace('\n', ' ').split('. ')
            except Exception as e:
                self.logger.warning(f"[Candidates picker] Page for id {name} is not found.")
        return result

    def get_candidates(self, claim: str) -> Dict:
        """
        The main method of the class that get the Wikipedia texts for related articles for given query
        :param claim: str query claim
        :return the dict with article names as keys and list of related sentences as values
        """

        candidates = self.get_wiki_candidates_NER(claim)
        self.logger.info(f"[Candidates picker] Candidates found: {', '.join(candidates)}")

        self.profiler.start_measure_local('wiki_texts')
        texts_dict = self.get_wiki_texts(candidates)
        self.profiler.finish_measure_local()

        return texts_dict
