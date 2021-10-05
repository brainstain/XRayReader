from tensorflow.examples.tutorials.mnist import input_data
import math
import model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image


def create_class_weight(labels_dict, mu=0.85):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight


class MNIST(model.TrainableData):

    def __init__(self, batch_size, data_dir='/tmp/data/MNIST_data/'):
        self.batch_size = batch_size
        self.mnist = input_data.read_data_sets(data_dir, one_hot=True)

    def data_generator(self):
        while 1:
            x_batch, y_batch = self.mnist.train.next_batch(batch_size=self.batch_size)
            yield ([x_batch, y_batch], [y_batch, x_batch])

    def get_next_test_batch(self):
        return self.mnist.test.next_batch(batch_size=self.batch_size)

    def get_next_train_batch(self):
        return self.mnist.train.next_batch(batch_size=self.batch_size)

    def get_all_test_data(self):
        return [self.mnist.test.images, self.mnist.test.labels]

    def get_data_element(self):
        return self.mnist


class XRay(model.TrainableData):

    def __init__(self, batch_size, data_dir='/home/michael/data/chestxray/',
                 image_dir='images256/',
                 test_size=0.2, verify_test_size=0.05,
                 image_size=(256,256), batches_buffer=20,
                 mutual_exclusive=True,
                 include_no_finding=True):
        self.batches_buffer = batches_buffer
        if mutual_exclusive:
            xray_data = pd.read_csv(data_dir + 'Data_Entry_Single_Finding_2017.csv')
        else:
            xray_data = pd.read_csv(data_dir + 'Data_Entry_2017.csv')
        if not include_no_finding:
            xray_data = xray_data[xray_data['Finding Labels'] != 'No Finding']
        dummy_labels = xray_data['Finding Labels'].str.get_dummies('|')
        if not include_no_finding:
            dummy_labels = dummy_labels.assign(Other=0)
        self.columns = dummy_labels.columns
        # Create class weights for data
        class_count = dummy_labels.sum().to_dict()
        class_weight = {}
        for index, (k, v) in enumerate(class_count.items()):
            class_weight[index] = v
        self.class_weight = create_class_weight(class_weight, .1)

        self.X_train, X_test, self.y_train, y_test = train_test_split(xray_data['Image Index'],
                                                                      dummy_labels,
                                                                      test_size=test_size)
        self.X_test, self.X_verify, self.y_test, self.y_verify = train_test_split(
            X_test,
            y_test,
            test_size=verify_test_size)
        self.image_dir = data_dir + image_dir
        self.batch_size = batch_size
        self.image_size=image_size
        self.training_size = len(self.X_train)
        self.X_train = np.array([self.image_dir + name for name in self.X_train])
        self.X_test = np.array([self.image_dir + name for name in self.X_test])
        self.X_verify = np.array([self.image_dir + name for name in self.X_verify])
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)
        self.y_verify = np.array(self.y_verify)
        print("Training data length: {}".format(len(self.y_train)))
        print("Testing data length: {}".format(len(self.y_test)))
        print("Verify data length: {}".format(len(self.y_verify)))

    def files_to_array(self, file_list, normalize=True):
        x = np.array([np.array(Image.open(fname)) for fname in file_list])
        if normalize:
            x = x / 255
        return x

    def data_generator(self, include_reverse=True):
        i = 0
        epoch_size = self.training_size // self.batch_size
        file_pointer = np.arange(self.training_size)
        np.random.shuffle(file_pointer)
        buffer = []
        start = 0
        while 1:
            if i >= epoch_size:  # start new epoch
                i = 0
                np.random.shuffle(file_pointer)

            if i % self.batches_buffer == 0:  # Load up the buffer
                buffer_end = i * self.batch_size + self.batches_buffer * self.batch_size
                file_names = self.X_train[
                    file_pointer[i * self.batch_size:buffer_end]]
                buffer = self.files_to_array(file_names, normalize=True)
                start = 0
            end = start + self.batch_size
            x_batch = buffer[start:end]
            y_batch = self.y_train[file_pointer[start:end]]
            if include_reverse:
                yield ([x_batch, y_batch], [y_batch, x_batch])
            else:
                yield [x_batch, y_batch]
            i += 1
            start += self.batch_size

    def get_next_test_batch(self):
        pass

    def get_next_train_batch(self):
        pass

    def get_all_test_data(self):
        test_images = self.files_to_array(self.X_test)
        return [test_images, self.y_test]

    def validation_generator(self):
        range_len = len(self.X_test) // self.batch_size
        i = 0
        while True:
            # Reset iterator index to 0 at end of range
            if i == range_len:
                i = 0
            yield [self.files_to_array(
                self.X_test[i*self.batch_size:(i+1)*self.batch_size]),
                self.y_test[i*self.batch_size:(i+1)*self.batch_size]]
            i += 1

    def get_all_validation_data(self):
        validation_images = self.files_to_array(self.X_verify)
        return [validation_images, self.y_verify]