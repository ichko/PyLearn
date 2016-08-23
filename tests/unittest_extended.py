import unittest


class TestCase(unittest.TestCase):

    def assertListsAlmostEqual(self, list_a, list_b, eps=0.01):
        result = all(abs(x - y) < eps for x, y in zip(list_a, list_b))
        if not result:
            print('list_a =', list_a, '\nlist_b =', list_b)

        self.assertTrue(result)
