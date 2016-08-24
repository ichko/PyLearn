from io import StringIO

from .base_test import BaseTest

from pylearn import data_loader


class TestDataLoader(BaseTest):

    def set_up_file_content(self, content):
        data_loader.open_file = lambda path: StringIO(content)

    def test_simple_matrix_parse(self):
        self.set_up_file_content("1,1,2\n2,2,3\n4,4,5")
        actual = data_loader.parse_data("file_name")
        self.assertListEqual([[1, 1], [2, 2], [4, 4]], actual[0])
        self.assertListEqual([2, 3, 5], actual[1])

    def test_complex_matrix_parse(self):
        self.set_up_file_content("1,1,3,6,2\n2,5,6,2,3\n4,4,5,5,6")
        actual = data_loader.parse_data("file_name")
        self.assertListEqual([[1, 1, 3, 6], [2, 5, 6, 2], [4, 4, 5, 5]],
                             actual[0])
        self.assertListEqual([2, 3, 6], actual[1])

    def test_trim_spaces(self):
        self.set_up_file_content("  2,3  ,2  \n 2,  5, 3\n4 ,7  ,6")
        actual = data_loader.parse_data("file_name")
        self.assertListEqual([[2, 3], [2, 5], [4, 7]], actual[0])
        self.assertListEqual([2, 3, 6], actual[1])


if __name__ == '__main__':
    unittest.main()
