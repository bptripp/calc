"""
This file is based on Keras example code.
"""

from __future__ import print_function
from calc.cnn_keras import make_model_from_network, subsample_maps, prune_maps, prune_connections
import keras
from keras.layers import Conv2D
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten
import os
import pickle


if __name__ == '__main__':
    with open('../../generated-files/calc-training-mini2.pickle', 'rb') as f:
        data = pickle.load(f)

    net = data['nets'][3] #25M, 39M, 53M, 7M, 35M
    #smallest (Trainable params: 92,336,653) (Trainable params: 13,399,251 with pruning)
    # Trainable params: 4,481,743 with pruning and subsampling
    # New network has 54,597,314 params after pruning and subsampling

    last_conv_layer = 'TEpd_5'
    # last_conv_layer = 'V4_4'

    # TODO: clip this in optimization?
    for connection in net.connections:
        if connection.w < 1:
            connection.w = 1

    # net.connections[16].w = 1.1 #this might be the bug, was 0.62
    # net.print()
    # assert False

    # # ******** get rid of most of the network to find bug
    # net.layers = net.layers[:10]
    # new_connections = net.connections[:6]
    # new_connections.extend(net.connections[10:17])  # 10:14 is one connection to V4_4
    # net.connections = new_connections
    # net.print()

    subsample_indices = subsample_maps(net)
    subsample_indices = prune_maps(net, subsample_indices, last_conv_layer)

    subsample_indices = prune_connections(net, subsample_indices)
    # for i in range(len(subsample_indices)):
    #     print('{}->{}: {}'.format(net.connections[i].pre.name, net.connections[i].post.name, len(subsample_indices[i])))

    # assert False

    input_layer = net.find_layer('INPUT')
    input_channels = int(input_layer.m)
    input_width = 32  # need this for CIFAR-10 (like looking through a hole)
    input = keras.Input(shape=(input_width, input_width, input_channels, ))

    # #TODO: experiment with Dropout here to work on overfitting
    output_layer = make_model_from_network(net, input, last_conv_layer, subsample_indices=subsample_indices)
    layer_f = Flatten()(output_layer)

    layer_d1 = Dense(128, activation='relu')(layer_f)
    layer_d2 = Dense(128, activation='relu')(layer_d1)
    layer_classifier = Dense(10, activation='softmax')(layer_d2)
    model = keras.Model(inputs=input, outputs=layer_classifier)

    # model.save('../../generated-files/calc-CIFAR-model.h5')
    #
    # model = keras.models.load_model('../../generated-files/calc-CIFAR-model.h5')

    # # add custom regularizers and constraints after loading, as Keras doesn't like to deserialize them
    # for layer in model.layers:
    #     if isinstance(layer, Conv2D):
    #         layer.kernel_regularizer = L1Control(l1=0.001)
    #         layer.kernel_constraint = SnapToZero()

    batch_size = 32
    num_classes = 10
    epochs = 100
    data_augmentation = True
    num_predictions = 20
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

    opt = keras.optimizers.adam(lr=.001)

    checkpoint_callback = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True,
                                    save_weights_only=False, mode='auto', period=1)

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

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
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
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
                                callbacks=[checkpoint_callback, early_stopping_callback])

            #TODO: actual targets
            l1s = []
            for layer in model.layers:
                if isinstance(layer, Conv2D):
                    l1 = model.layers[0].kernel_regularizer.control(model.layers[0])
                    l1s.append(l1)
            print('L1 weights:')
            print(l1s)

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
