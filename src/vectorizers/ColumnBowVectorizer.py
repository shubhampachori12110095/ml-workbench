from sets import Set
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin
from toolz.functoolz import pipe, partial
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

    def apply_stopwords_filter_to_tokenlist(self, tokenlist):
        return [self._apply_stopwords_filter(tokens) for tokens in tokenlist]

    def _get_ngrams(self, tokens, n):
        return [''.join(tokens[0:i]) for i in n if i <= len(tokens)] 

    def _build_vocabulary(self, tokenlist, ngrams):
        return reduce(concat, 
            [self._get_ngrams(tokens, ngrams) for tokens in tokenlist])

    def _build_dictionary(self, vocabulary):
        return { word:index for index, word in enumerate(vocabulary) }

    def fit(self, df, *_):
        tokenlist = pipe(
            df,
            self._get_df_column_row_values,
            self._tokenize_row_value,
            self.apply_stopwords_filter_to_tokenlist
        )

        self._vocabulary = self._build_vocabulary(tokenlist, ngrams=[1,2,3])
        self._dictionary = self._build_dictionary(self._vocabulary)

        return self

    def transform(self, df):
        """
        for each tokenised sentence, convert that into arrays 
        of which words occurred, where the order of the occurrence
        matches that of the vocabulary.
        """
        column_vectors = []

        for column in df:
            rows = df[column].values
            column_vector = []

            for row in rows:
                tokens = word_tokenize(row)
                row_vector = dict.fromkeys(
                    range(0, len(self._vocabulary)), 0)

                tokensFiltered = []
                for token in tokens:
                    if token not in self._stopwords:
                        tokensFiltered.append(token)

                for index, token in enumerate(tokensFiltered):
                    try: 
                        # 1-grams
                        dict_key = self._dictionary[token]
                        if dict_key is not -1:
                            row_vector[dict_key] = 1

                        # # 2-grams
                        if (index > 0 and index < len(tokensFiltered)):
                            ngram = tokensFiltered[index-1] + tokensFiltered[index]
                            dict_key = self._dictionary[ngram]
                            if dict_key is not -1:
                                row_vector[dict_key] = 1

                        # 3-grams
                        if (index > 1 and index < len(tokensFiltered)):
                            ngram = tokensFiltered[index-2] + tokensFiltered[index-1] + tokensFiltered[index]
                            dict_key = self._dictionary[ngram]
                            if dict_key is not -1:
                                row_vector[dict_key] = 1

                    except: 
                        pass
                    
                column_vector.append(
                    list(row_vector.values())
                )
                
            column_vectors.append(column_vector)

        # convert to feature representation
        num_data_rows = len(column_vectors[0])
        feature_vectors = []

        for index in range(0, num_data_rows):
            feature_vector = []

            for column in column_vectors:
                feature_vector += column[index]

            feature_vectors.append(feature_vector)

        return np.stack(feature_vectors)
