from sets import Set
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin
from toolz.functoolz import pipe
from toolz.dicttoolz import merge
from toolz.itertoolz import unique
from operator import concat

class ColumnBowVectorizer(TransformerMixin):
    def __init__(self):
        self._stopwords = stopwords.words('english') + \
            ['.', '?'] + ['went', 'moved', 'travelled', 'journeyed', 'back']

    def _get_df_column_row_values(self, df):
        return df.values.flatten()

    def _tokenize_row_value(self, values):
        return map(word_tokenize, values)

    def _apply_stopwords_filter(self, tokens):
        return [token for token in tokens if token not in self._stopwords]

    def _apply_stopwords_filter_to_tokenlist(self, tokenlist):
        return [self._apply_stopwords_filter(tokens) for tokens in tokenlist]

    def _get_ngrams(self, tokens, n):
        return unique(
            [''.join(tokens[0:i]) for i in n if i <= len(tokens)] + tokens
        )

    def _preprocess_row(self, values):
        return pipe(
            values,
            self._tokenize_row_value,
            self._apply_stopwords_filter_to_tokenlist
        )

    def _build_vocabulary(self, tokenlist, ngrams):
        return reduce(concat, 
            [list(self._get_ngrams(tokens, ngrams)) for tokens in tokenlist])

    def _build_dictionary(self, vocabulary):
        return { word:index for index, word in enumerate(vocabulary) }

    def fit(self, df, *_):
        """
        build a dictionary from the given phrases that will be used to 
        transform phrases into bag of word arrays representing that phrase.
        """
        tokenlist = pipe(
            df,
            self._get_df_column_row_values,
            self._preprocess_row
        )
        self._vocabulary = self._build_vocabulary(tokenlist, ngrams=[1,2,3])
        self._dictionary = self._build_dictionary(self._vocabulary)
        return self

    def _get_dictionary_index(self, ngram):
        dict_key = self._dictionary[ngram]
        if dict_key is not -1:
            return dict_key

    def _get_row_indices(self, row_ngrams):
        return [self._get_dictionary_index(ngram) for row in row_ngrams for ngram in row]

    def _create_row_vector(self, tokenlist, ngrams):
        row_indices = pipe(
            [self._get_ngrams(tokenlist, ngrams)],
            self._get_row_indices,
        )
        return merge(
            dict.fromkeys(range(0, len(self._vocabulary)), 0),
            dict.fromkeys(row_indices, 1),
        )

    def _vectorize_feature(self, tokenlist, ngrams):
        return [self._create_row_vector(tokens, ngrams) for tokens in tokenlist]

    def transform(self, df):
        """
        transform given phrases into bag of word vectors.
        """
        tokenized_rows = [self._preprocess_row(value) for value in df.values],
        return [self._vectorize_feature(feature, [1,2,3]) 
            for row in tokenized_rows for feature in row]