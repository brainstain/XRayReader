import unittest
import datasets as ds


class MyTestCase(unittest.TestCase):
    def test_generator(self):
        data = ds.XRay(50)
        ([x_batch, y_batch], [_, _]) = data.data_generator().__next__()
        self.assertEqual(x_batch.shape, (50, 256, 256))
        self.assertEqual(y_batch.shape, (50, 15))


if __name__ == '__main__':
    unittest.main()
