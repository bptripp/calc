"""
This file is based on Keras example code.
"""

from __future__ import print_function
import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())
from argparse import ArgumentParser

from calc.cnn_keras import make_model_from_network, subsample_maps, snip
from calc.cnn_keras import prune_maps, prune_connections, prune_layers, SparsityConstraint
import keras
from keras import backend as K
from keras.layers import Conv2D
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
import os
import pickle


model_file = 'calc-CIFAR-model.h5'


def build_model(optimization_result_file, last_conv_layer, save=True, c_scale=1, sigma_scale=1):
    with open(optimization_result_file, 'rb') as file:
        data = pickle.load(file)

    net = data['net']

    for layer in net.layers:
        layer.m = int(round(layer.m))

    def set_c(connection, new_c):
        connection.sigma = connection.sigma * connection.c / new_c
        connection.c = new_c

    for connection in net.connections:
        if 'INPUT' in connection.pre.name or 'LGN' in connection.pre.name:
            set_c(connection, 1)

    alpha_total = 0
    beta_total = 0
    for connection in net.connections:
        if connection.pre.name == 'V1_4Calpha':
            alpha_total += connection.c * connection.sigma
        if connection.pre.name == 'V1_4Cbeta':
            beta_total += connection.c * connection.sigma

    for connection in net.connections:
        if connection.pre.name == 'V1_4Calpha':
            fraction = connection.c * connection.sigma / alpha_total
            set_c(connection, fraction)
        if connection.pre.name == 'V1_4Cbeta':
            fraction = connection.c * connection.sigma / beta_total
            set_c(connection, fraction)

    # net.print()
    # assert False

    net.scale_c(c_scale)
    net.scale_sigma(sigma_scale)

    subsample_indices = subsample_maps(net)
    subsample_indices = prune_maps(net, subsample_indices, last_conv_layer)
    subsample_indices = prune_connections(net, subsample_indices)
    subsample_indices = prune_layers(net, subsample_indices)

    # keep track of subsample indices by name, because connection indices may change
    subsample_map = {}
    for i in range(len(net.connections)):
        subsample_map[net.connections[i].get_name()] = subsample_indices[i]

    # for i in range(len(net.layers)):
    #     print('{} subsamples: {}'.format(net.layers[i].name, subsample_indices[i]))

    # print(len(net.connections))
    # print(len(subsample_indices))
    for i in range(len(subsample_indices)):
        print('{}->{}: {}'.format(net.connections[i].pre.name, net.connections[i].post.name, len(subsample_indices[i])))

    removed_indices = net.prune_dead_ends([last_conv_layer])
    removed_indices = np.sort(removed_indices)
    for i in range(len(removed_indices)):
        next_largest_removed_index = removed_indices[-1-i]
        del subsample_indices[next_largest_removed_index]

    subsample_indices = []
    for i in range(len(net.connections)):
        subsample_indices.append(subsample_map[net.connections[i].get_name()])

    # print('**********')
    # print(len(net.connections))
    # print(len(subsample_indices))
    for i in range(len(subsample_indices)):
        print('{}->{}: {}'.format(net.connections[i].pre.name, net.connections[i].post.name, len(subsample_indices[i])))

    for layer in net.layers:
        if layer.m < 1:
            print('{} has {} maps'.format(layer.name, layer.m))

    for connection in net.connections:
        if connection.w < 1:
            print('setting w to 1 for {}->{}'.format(connection.pre.name, connection.post.name))
            connection.w = 1

    net.print()
    # assert False

    input_layer = net.find_layer('INPUT')
    # input_channels = int(input_layer.m)
    input_channels = input_layer.m
    input_width = 32  # need this for CIFAR-10 (like looking through a hole)
    input = keras.Input(shape=(input_width, input_width, input_channels, ))

    output_layer, sparse_layers = make_model_from_network(net, input, last_conv_layer, subsample_indices=subsample_indices)

    x = Conv2D(100, (1,1))(output_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=input, outputs=x)

    if save:
        model.save(model_file)

    return model, sparse_layers


def load_model():
    from calc.cnn_keras import SparsityConstraint
    return keras.models.load_model(model_file, custom_objects={'SparsityConstraint': SparsityConstraint})
    # return keras.models.load_model(model_file)


def restore_weights(model, filepath):
    model.load_weights(filepath, by_name=True)


def restore_constraints(model, filepath):
    with open(filepath, 'rb') as file:
        kernel_masks = pickle.load(file)

    for layer in model.layers:
        if isinstance(layer, Conv2D) and isinstance(layer.kernel_constraint, SparsityConstraint):
            layer.kernel_constraint.non_zero = kernel_masks[layer.name]
            layer.kernel_constraint.mask = K.constant(1 * layer.kernel_constraint.non_zero)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c_scale", dest="c_scale", default=1,
                        help="scale factor for channel-wise sparsities in log space")
    parser.add_argument("-sigma_scale", dest="sigma_scale", default=1,
                        help="scale factor for element-wise sparsities in log space")

    args = parser.parse_args()
    c_scale = float(args.c_scale)
    sigma_scale = float(args.sigma_scale)
    print('log-sparsity scale factors: c_scale={} sigma_scale={}'.format(c_scale, sigma_scale))

    model, sparse_layers = build_model('calc/optimization-result-PITv.pkl', 'PITv_2/3',
                                       save=True, c_scale=c_scale, sigma_scale=sigma_scale)

    # model = load_model()
    # restore_constraints(model, 'kernel_masks.pkl')
    # restore_weights(model, 'weights.01-0.78.hdf5')

    batch_size = 32
    num_classes = 10
    epochs = 1000
    data_augmentation = True
    # num_predictions = 20
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_cifar10_trained_model.h5'

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    opt = keras.optimizers.adam(lr=.01)

    checkpoint_callback = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True,
                                    save_weights_only=False, mode='auto', period=1)

    # early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    snip(sparse_layers, model, x_test[:128,:,:,:], y_test[:128,:])

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=True,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=True,  # apply ZCA whitening
            rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # l1 = model.layers[0].kernel_regularizer.control(model.layers[0], target, threshold=threshold)

        # Fit the model on the batches generated by datagen.flow().
        for i in range(epochs):
            model.fit_generator(datagen.flow(x_train, y_train,
                                             batch_size=batch_size),
                                epochs=1,
                                validation_data=(x_test, y_test),
                                workers=4,
                                # callbacks=[checkpoint_callback, early_stopping_callback],
                                callbacks=[checkpoint_callback],
                                steps_per_epoch=x_train.shape[0]/batch_size)


    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
