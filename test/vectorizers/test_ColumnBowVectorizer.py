import unittest
import snapshottest
import pandas as pd
import json
from src.vectorizers.ColumnBowVectorizer import ColumnBowVectorizer

class TestColumnBowVectorizer(snapshottest.TestCase):

    def setUp(self):
        self.df = pd.read_json('{"fact1":{"0":"John travelled to the hallway.","1":"Daniel went back to the bathroom.","2":"John went to the hallway.","3":"Sandra travelled to the hallway.","4":"Sandra went back to the bathroom."},"fact2":{"0":"Mary journeyed to the bathroom.","1":"John moved to the bedroom.","2":"Sandra journeyed to the kitchen.","3":"John went to the garden.","4":"Sandra moved to the kitchen."},"question":{"0":"Where is John? ","1":"Where is Mary? ","2":"Where is Sandra? ","3":"Where is Sandra? ","4":"Where is Sandra? "}}')

    def test_vectorizes_dataframe(self):
        vectorizer = ColumnBowVectorizer()
        vectorizer.fit(self.df)

        self.assertMatchSnapshot(json.dumps(list(vectorizer._vocabulary)))
        self.assertMatchSnapshot(json.dumps(vectorizer._dictionary))

        feature_vectors = vectorizer.transform(self.df)
        self.assertMatchSnapshot(json.dumps(feature_vectors))

if __name__ == '__main__':
    unittest.main()