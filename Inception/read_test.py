import unittest
import Inception.read_xray as xr


class InceptionTest(unittest.TestCase):
    def test_reading(self):
        weightpath = '/home/michael/logs/xray/weights/trained_model_2.h5'
        reader = xr.XrayReader(weightpath)
        img = '/home/michael/data/chestxray/images299/00000008_000.png' #Cardiomegaly
        print("Cardiomegaly")
        print(reader.predict(img))
        img = '/home/michael/data/chestxray/images299/00000013_011.png'
        print("Pneumothorax")
        print(reader.predict(img))
        img = '/home/michael/data/chestxray/images299/00000013_046.png'
        print("Infiltration")
        print(reader.predict(img))


if __name__ == '__main__':
    unittest.main()
