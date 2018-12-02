import unittest
import snapshottest
from src.data_loaders.BablTaskLoader import parse_data

class BablTaskLoaderTest(snapshottest.TestCase):

    def test_converts_task1_data_to_dataframe(self):
        df = parse_data('test_data/babl_task1_excerpt.txt')
        self.assertMatchSnapshot(df.to_json())

if __name__ == '__main__':
    unittest.main()