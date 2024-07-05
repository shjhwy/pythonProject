import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        super().__init__()
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
