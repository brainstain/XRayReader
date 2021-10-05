from CapsNet import capsnet as cn
import datasets
import os
import pickle

if __name__ == "__main__":
    working_directory = '/tmp/models/mnistcaps/'

    batch_size = 100
    data = datasets.MNIST(batch_size=batch_size)

    capsnet = cn.CapsNet(input_shape=[784], n_class=10, reshape=[28, 28, 1],
                         save_dir=working_directory)

    capsnet.train_model.summary()
    if not os.path.exists(working_directory + 'weights'):
        os.makedirs(working_directory + 'weights/')
    else:  # if there are weights load them
        if os.path.isfile(working_directory + 'weights/trained_model.h5'):
            capsnet.load_weights(working_directory + 'weights/trained_model.h5')
        if os.path.isfile(working_directory + 'weights/optimizer.pkl'):
            with open(working_directory + 'weights/optimizer.pkl', 'rb') as input:
                capsnet.load_optimizer_weights(pickle.load(input))

    model = capsnet.train(data, batch_size, 55000 // batch_size, epochs=400)

    model.save_weights(working_directory + 'weights/trained_model.h5', overwrite=True)
    weights = model.optimizer.get_weights()

    with open(working_directory + 'weights/optimizer.pkl', 'wb') as output:
        pickle.dump(weights, output, pickle.HIGHEST_PROTOCOL)

    print('Trained model saved to \'%sweights/trained_model.h5\'' % working_directory)
    capsnet.test(data)
