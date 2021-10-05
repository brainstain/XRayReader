from functools import partial

from keras.applications.inception_v3 import InceptionV3
import datasets
from keras import callbacks
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD
import Inception.inception_utils as utils

save_dir = '/home/michael/logs/xray/'

# create the base pre-trained model
base_model = InceptionV3(weights=None, include_top=False, input_shape=(299, 299, 3))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- 15 possible xray outcomes including no finding
predictions = Dense(15, activation='sigmoid', name='predict')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

steps_per_epoch = 1034
batch_size = 50

# Loads previously trained base weights
# model.load_weights(save_dir + 'weights/base_model_2.h5')

# compile the model (should be done *after* setting layers to non-trainable)
loss_func = partial(utils.weighted_binary_crossentropy, pos_weight=2)

model.compile(optimizer='adam', loss=loss_func,
              metrics=['binary_accuracy', utils.correct_positive, utils.false_positive, utils.false_negative])


log = callbacks.CSVLogger(save_dir + 'log.csv')
tb = callbacks.TensorBoard(log_dir=save_dir + 'tensorboard-logs',
                           batch_size=batch_size)
checkpoint = callbacks.ModelCheckpoint(save_dir + '/checkpoint/weights-{epoch:02d}.h5',
                                       monitor='val_predict_acc',
                                       save_best_only=True,
                                       save_weights_only=True,
                                       verbose=1)

data = datasets.XRay(batch_size=batch_size,
                     batches_buffer=10,
                     image_dir='images299/',
                     mutual_exclusive=False,
                     include_no_finding=False)

# train the model on the new data for a few epochs
model.fit_generator(generator=data.data_generator(False),
                    epochs=30,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=data.validation_generator(),
                    validation_steps=15,
                    callbacks=[log, tb, checkpoint],
                    class_weight=data.class_weight)

model.save_weights(save_dir + 'weights/base_model_scratch.h5', overwrite=True)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
# for layer in model.layers[:249]:
#    layer.trainable = False
# for layer in model.layers[249:]:
#    layer.trainable = True


model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss=loss_func,
              metrics=['binary_accuracy', utils.correct_positive, utils.false_positive, utils.false_negative])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(generator=data.data_generator(False),
                    epochs=50,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=data.validation_generator(),
                    validation_steps=15,
                    callbacks=[log, tb, checkpoint],
                    class_weight=data.class_weight)
model.save_weights(save_dir + 'weights/trained_model_check_scratch.h5', overwrite=True)
# model.save(save_dir + '/model/trained_model.h5')

model.fit_generator(generator=data.data_generator(False),
                    epochs=100,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=data.validation_generator(),
                    validation_steps=15,
                    callbacks=[log, tb, checkpoint],
                    class_weight=data.class_weight)
model.save_weights(save_dir + 'weights/trained_model_deeper_scratch.h5', overwrite=True)

# weights/trained_model.h5 loss - binary_crossentropy and sigmoid

