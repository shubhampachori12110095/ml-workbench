import unittest
import snapshottest
from src.data_loaders.BablTask1Loader import parse_data

class BablTask1LoaderTest(snapshottest.TestCase):

    def test_converts_data_to_dataframe(self):
        df = parse_data('test_data/babl_task1_excerpt.txt')
        self.assertMatchSnapshot(df.to_json())

if __name__ == '__main__':
    unittest.main()