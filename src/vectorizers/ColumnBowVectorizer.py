import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from toolz.functoolz import pipe, partial
from toolz.dicttoolz import merge
from toolz.itertoolz import unique, mapcat
from itertools import chain

class ColumnBowVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, ngrams=[1,2,3]):
        self._stopwords = stopwords.words('english') + \
            ['.', '?'] + ['went', 'moved', 'travelled', 'journeyed', 'back']
        self.ngrams=ngrams

    def _get_df_column_row_values(self, df):
        return df.values.flatten()

    def _tokenize_row_value(self, values):
        return map(word_tokenize, values)

    def _apply_stopwords_filter(self, tokens):
        return [token for token in tokens if token not in self._stopwords]

    def _apply_stopwords_filter_to_tokenlist(self, tokenlist):
        return [self._apply_stopwords_filter(tokens) for tokens in tokenlist]

    def _get_ngrams(self, tokens, ngram_sizes):
        ngram_joins = [''.join(tokens[:ngram_size]) for ngram_size in ngram_sizes if ngram_size <= len(tokens)]
        return unique(
            ngram_joins + tokens
        )

    def _preprocess_row(self, values):
        return pipe(
            values,
            self._tokenize_row_value,
            self._apply_stopwords_filter_to_tokenlist
        )

    def _build_vocabulary(self, tokenlist, ngrams):
        return pipe(
            (self._get_ngrams(tokens, ngrams) for tokens in tokenlist),
            chain.from_iterable,
            list
        )

    def _build_dictionary(self, vocabulary):
        return { word:index for index, word in enumerate(vocabulary) }

    def fit(self, X, y=None):
        """
        build a dictionary from the given phrases that will be used to 
        transform phrases into bag of word arrays representing that phrase.
        """

        tokenlist = pipe(
            X,
            self._get_df_column_row_values,
            self._preprocess_row
        )
        self._vocabulary = self._build_vocabulary(tokenlist, ngrams=self.ngrams)
        self._dictionary = self._build_dictionary(self._vocabulary)
        return self

    def _get_dictionary_index(self, ngram):
        dict_key = self._dictionary[ngram]
        if dict_key is not -1:
            return dict_key

    def _get_row_indices(self, row_ngrams):
        return [self._get_dictionary_index(ngram) for row in row_ngrams for ngram in row]

    def _get_empty_vector(self, size):
        return [0] * size

    def _create_feature_vector(self, tokens, ngrams):
        row_indices = pipe(
            [self._get_ngrams(tokens, ngrams)],
            self._get_row_indices,
        )
        vector = self._get_empty_vector(len(self._vocabulary))
        for index in row_indices:
            vector[index] = 1
        return vector

    def _vectorize_feature(self, tokens, ngrams):
        return pipe(
            (self._create_feature_vector(token, ngrams) for token in tokens),
            chain.from_iterable,
            list
        )

    def transform(self, X):
        """
        transform given phrases into bag of word vectors.
        """
        return pipe(
            [self._preprocess_row(value) for value in X.values],
            partial(map, lambda row: row),
            partial(
                map, 
                partial(self._vectorize_feature, ngrams=self.ngrams)
            )
        )