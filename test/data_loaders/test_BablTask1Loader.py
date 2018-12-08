import unittest
import snapshottest
from src.data_loaders.BablTask1Loader import parse_data, split_parse_data

class BablTask1LoaderTest(snapshottest.TestCase):

    def test_converts_data_to_dataframe(self):
        df = parse_data('test_data/babl_task1_excerpt.txt')
        self.assertMatchSnapshot(df.to_json())

    def test_splits_dataframe_into_X_and_y(self):
        X, y = split_parse_data('test_data/babl_task1_excerpt.txt')
        self.assertMatchSnapshot(X.to_json())
        self.assertMatchSnapshot(y.to_json())

if __name__ == '__main__':
    unittest.main()