from CapsNet import capsnet as cn
import datasets
import os
import pickle
from keras import backend as K

if 'tensorflow' == K.backend():
    import tensorflow as tf

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = K.tensorflow_backend.get_session()
    K.tensorflow_backend.set_session(tf.Session(config=config))

if __name__ == "__main__":
    working_directory = '/tmp/models/xray/'

    batch_size = 10
    data = datasets.XRay(batch_size=batch_size, batches_buffer=10,image_dir='images128/')

    capsnet = cn.CapsNet(input_shape=[128, 128], n_class=15, reshape=[128, 128, 1],
                         save_dir=working_directory, digit_caps_dim=16, conv_filters=256,
                         n_channels=32, kernal_size=14)

    capsnet.train_model.summary()
    if not os.path.exists(working_directory + 'weights'):
        os.makedirs(working_directory + 'weights/')
    else:  # if there are weights load them
        if os.path.isfile(working_directory + 'weights/trained_model.h5'):
            capsnet.load_weights(working_directory + 'weights/trained_model.h5')
        if os.path.isfile(working_directory + 'weights/optimizer.pkl'):
            with open(working_directory + 'weights/optimizer.pkl', 'rb') as input:
                capsnet.load_optimizer_weights(pickle.load(input))

    model = capsnet.train(data, batch_size, 896, epochs=20)

    model.save_weights(working_directory + 'weights/trained_model.h5', overwrite=True)
    weights = model.optimizer.get_weights()

    with open(working_directory + 'weights/optimizer.pkl', 'wb') as output:
        pickle.dump(weights, output, pickle.HIGHEST_PROTOCOL)

    print('Trained model saved to \'%sweights/trained_model.h5\'' % working_directory)
    capsnet.test(data)
