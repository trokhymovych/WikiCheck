import logging
import wikipedia
from flair.data import Sentence
from flair.models import SequenceTagger

logger = logging.getLogger("WIKISELECTOR")

class WikiCandidatesSelector:
    """
    Class responsible for model of candidates selection from Wikipedia
    :param logger: logger to use in model
    """

    def __init__(self, logger=logger, **kwargs):
        # load the NER tagger
        self.logger = logger
        self.tagger = SequenceTagger.load('ner-fast')  # 'ner-fast'
        self.separate = True
        self.n = 3
        self.logger.info("Flair NER model is loaded and ready to use.")

    @staticmethod
    def getting_wiki_candidates_raw(query, n=10):
        """
        Wikipedia query
        """
        search_results = wikipedia.search(query, results=n)
        return [t.replace(' ', '_') for t in search_results]

    def get_entities_flair(self, text):
        """
        Get the list of named entities for given text.
        :param text: str, text for NER extraction
        """
        # make and process sentence
        sentence = Sentence(text)
        self.tagger.predict(sentence)
        ents = []
        # iterate over entities and print
        for entity in sentence.get_spans('ner'):
            ents.append(entity.text)
        return ents

    def getting_wiki_candidates_NER(self, query):
        """
        Method to get the wikipedia articles candidates
        :param query: str query claim
        """

        ents = self.get_entities_flair(query)

        search_results = self.getting_wiki_candidates_raw(query, n=self.n)

        if not self.separate:
            search_results_en = self.getting_wiki_candidates_raw(' '.join(ents), n=self.n)

        else:
            search_results_en = []
            for e in ents:
                search_results_en += self.getting_wiki_candidates_raw(e, n=self.n)

        return set([t for t in search_results + search_results_en])

    @staticmethod
    def get_wiki_texts(articles_names):
        """
        Method that gets Wikipedia texts for given articles names
        """
        result = {}
        for name in articles_names:
            page = wikipedia.page(name)
            result[name] = page.content.split('. ')
        return result

    def get_candidates(self, claim):
        """
        The main method of the class that get the texts for given query

        """

        candidates = self.getting_wiki_candidates_NER(claim)
        texts_dict = self.get_wiki_texts(candidates)

        return texts_dict
