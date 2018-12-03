from sets import Set
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class ColumnBowVectorizer(object):
    def __init__(self):
        self._stopwords = stopwords.words('english')
        self._vocabulary = Set()
        self._dictionary = {}

    def fit(self, df):
        for column in df:
            rows = df[column].values

            for row in rows:
                tokens = word_tokenize(row)
                tokensFiltered = []

                for token in tokens:
                    if token not in self._stopwords:
                        tokensFiltered.append(token)

                [self._vocabulary.add(token) for token in tokensFiltered]
                # TODO: 2-grams & 3-grams

        self._dictionary = {
            x:index-1 for index, x in enumerate(self._vocabulary)
        }

    def transform(self, df):

        # for each tokenised sentence, convert that into arrays 
        # of which words occurred, where the order of the occurrence
        # matches that of the vocabulary
        column_vectors = []

        for column in df:
            rows = df[column].values
            column_vector = []

            for row in rows:
                tokens = word_tokenize(row)
                row_vector = dict.fromkeys(
                    range(0, len(self._vocabulary)), 
                    0
                )

                for token in tokens:
                    if token not in self._stopwords:
                        dict_key = self._dictionary[token]
                        if dict_key is not -1:
                            row_vector[dict_key] = 1
                
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