import json

from PIL import Image
import numpy as np
from keras import Model
from keras.applications import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense


def image_convert(file):
    raw_image = Image.open(file)

    # Check if image is 3 channel, convert if not
    if raw_image.mode != 'RGB':
        raw_image = raw_image.convert('RGB')

    im_array = np.array(raw_image)

    # check sizes
    shape = im_array.shape
    # square
    if (shape[0] != shape[1]) or (shape[0] != 299):
        raw_image = raw_image.resize((299,299), resample=Image.BICUBIC)
        im_array = np.array(raw_image)

    # normalize image
    im_array = im_array / 255

    return im_array


class XrayReader:
    def __init__(self, weights_path, decode_path='/home/michael/data/chestxray/data_labels_no_finding.json'):
        # Build out model from training - TODO: Move this to a standard place
        base_model = InceptionV3(weights=None, include_top=False, input_shape=(299, 299, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- 15 possible xray outcomes including no finding
        predictions = Dense(15, activation='sigmoid', name='predict')(x)

        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.load_weights(weights_path)

        with open(decode_path, 'r') as infile:
            self.decoding = json.load(infile)

    def predict(self, image, top=5):
        im_arr = np.array([image_convert(image)])
        predictions = self.model.predict(im_arr)
        results = []
        for prediction in predictions:
            top_indices = prediction.argsort()[-top:][::-1]
            result = [(self.decoding[i], prediction[i]) for i in top_indices]
            results.append(result)
        return results
