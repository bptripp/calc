"""
Experimenting with different methods of controlling sparsity in simple networks.

This code is derived from Keras example code:
https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py

The Keras license follows:
The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

# TODO: gain may affect performance (low gain may give weights more time to adapt)

# TODO: histogram of weights, see where zero is -- nothing definite but ~10^-3
# TODO: method to recompile model with different L1 weight

# TODO: check weight hist on first two weight files, shouldn't be sparse because I used null update - check
# TODO: run again with non-null update

import os
import numpy as np
import pickle
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import Regularizer
import keras.backend as K
from keras.constraints import Constraint


class SnapToZero(Constraint):
    """
    TODO: docs, threshold param
    TODO: not sure positive is OK; gradient update applied first but may be lots of zeros due to L1
    """

    def __call__(self, w):
        small = K.cast(K.less_equal(K.abs(w), 1e-4), K.floatx())
        lucky = K.cast(K.random_uniform(w.shape, 0, 1, K.floatx()) < 1e-4, K.floatx())
        positive = K.cast(K.greater(w, 0.0), K.floatx())

        w *= (1.0 - small)
        w += (1.1e-4 * lucky * small * positive)
        w -= (1.1e-4 * lucky * small * (1.0 - positive))

        return w


class L1Control(Regularizer):
    """
    Regularizer for L1 regularization with feedback control. The controlled variable
    is the l1 gain and the observed variable is the fraction of weights smaller than
    a threshold. To improve linearity of the plant, the controller actually acts on
    the exponent of the l1 gain. Specifically we take l1 = 10.^(-u-5) - 10.^(u-5),
    with u constrained in [-5 5]. This allows both positive and negative l1 weights,
    the latter to rapidly decrease sparsity.

    # Arguments
        l1: Float; L1 regularization factor.
    """

    def __init__(self, l1=0.0, pid_gains=[5, .5, 0], target=None):
        self.l1 = K.variable(K.cast_to_floatx(l1))
        self.pid_gains = pid_gains
        self.target = target

        self.error_integral = 0
        self.last_error = None

    def control(self, layer, target=None, threshold=1e-4):
        y = self._get_sparse_fraction(layer, threshold=threshold)

        if target is None:
            target = self.target

        error = y - target
        if self.last_error is not None:
            error_derivative = error - self.last_error
        else:
            error_derivative = 0

        u = - self.pid_gains[0]*error - self.pid_gains[1]*self.error_integral - self.pid_gains[2]*error_derivative
        new_l1 = 10**(-u-5) - 10**(u-5)
        new_l1 = np.clip(new_l1, -.2, .2)
        if new_l1 < 0:
            new_l1 = new_l1 / 10000

        print('control error: {} integral: {} derivative: {} u: {} l1: {}'.format(error, self.error_integral, error_derivative, u, new_l1))
        self.update(new_l1=new_l1)

        self.error_integral += error
        self.last_error = error

        return new_l1


    def _get_sparse_fraction(self, layer, threshold=1e-4):
        kernel = layer.get_weights()[0]
        abs_weights = abs(kernel.flatten())
        return (abs_weights > threshold).sum() / abs_weights.size

    def update(self, new_l1=0.0):
        update = K.update(self.l1, K.cast_to_floatx(new_l1))
        update.eval(session=K.get_session())

    def __call__(self, x):
        delta = K.constant(1e-5)
        beta = 1000
        alpha = 2.1
        regularization = K.sum(self.l1 * K.abs(x))
        # regularization = self.l1 * delta * K.sum((K.sqrt(1.0 + K.square(x/delta)) - 1.0))
        # regularization = self.l1 * delta * beta * alpha * K.exp(-K.square(K.log(K.abs(x)/beta/delta)))
        # regularization = self.l1 * delta * K.sum(
        #     (K.sqrt(1.0 + K.square(x/delta)) - 1.0) +
        #     beta * alpha * K.exp(-K.square(K.log(K.abs(x) / beta / delta)))
        #     )
        return regularization

    def get_config(self):
        return {'l1': float(K.cast_to_floatx(self.l1.eval(session=K.get_session())))}


def get_model():
    model = keras.Sequential()
    model.add(Conv2D(32, (5,5), input_shape=(32,32,3), strides=(1, 1), padding='valid', activation='relu',
                     kernel_regularizer=L1Control(l1=0.01), kernel_constraint=SnapToZero()))
    model.add(Conv2D(32, (3,3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(32, (3,3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # opt = BrAdam(lr=.001)
    opt = keras.optimizers.adam(lr=.001)
    # opt = keras.optimizers.adam(lr=.001, amsgrad=True)
    # opt = keras.optimizers.adam(lr=.001, beta_1=0.99, amsgrad=True)


    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def show_weights(model):
    kernel = model.layers[0].get_weights()[0]
    abs_weights = abs(kernel.flatten())
    print('min: {} max: {}'.format(np.min(abs_weights), np.max(abs_weights)))
    import matplotlib.pyplot as plt
    plt.hist(abs_weights, 5000, cumulative=True)
    plt.gca().set_xscale('log')
    plt.show()


def sparsity_dynamics_negative():
    fractions = []
    f1 = sparsity_dynamics_step(model, x_train, y_train, l1=.01)
    fractions.extend(f1)
    f2 = sparsity_dynamics_step(model, x_train, y_train, l1=-.001)
    fractions.extend(f2)
    f3 = sparsity_dynamics_step(model, x_train, y_train, l1=.01)
    fractions.extend(f3)


def control_experiment(model, x_train, y_train):
    fractions = []
    l1s = []

    for i in range(200):
        f, l1 = sparsity_control_step(model, x_train, y_train, target=.5, epochs=.14)

        fractions.extend(f)
        l1s.append(l1)

    # show_weights(model)
    return fractions, l1s


def sparsity_rebound_experiment(model, x_train, y_train, x_test, y_test):
    fractions = []
    scores = []

    fractions.append(model.layers[0].kernel_regularizer._get_sparse_fraction(model.layers[0]))
    scores.append(model.evaluate(x_test, y_test, verbose=1))

    for i in range(20):
        fractions.extend(sparsity_dynamics_step(model, x_train, y_train, l1=.001, epochs=1))
        scores.append(model.evaluate(x_test, y_test, verbose=1))

    # for i in range(8):
    #     fractions.extend(sparsity_dynamics_step(model, x_train, y_train, l1=.0, epochs=1))
    #     scores.append(model.evaluate(x_test, y_test, verbose=1))

    # for i in range(8):
    #     fractions.extend(sparsity_dynamics_step(model, x_train, y_train, l1=-.001, epochs=1))
    #     scores.append(model.evaluate(x_test, y_test, verbose=1))

    for i in range(4):
        fractions.extend(sparsity_dynamics_step(model, x_train, y_train, l1=-.1, epochs=1))
        scores.append(model.evaluate(x_test, y_test, verbose=1))

    return fractions, scores


def sparsity_nonlinearity_experiment(x_train, y_train, x_test, y_test, reps=1):
    # train with different L1 costs on fresh models

    # l1s = [1]
    # l1s = [-.01, -.001, -.0001, .0001, .001, .01, .1, 1]
    l1s = [-.0001, 0., .0001, .001, .01, .1, 1]
    # l1s = [-.01, 1]

    fractions = []
    scores = []
    for l1 in l1s:
        print('l1: {}'.format(l1))
        f = []
        s = []
        for i in range(reps):
            model = get_model()
            fi = []
            fi.append(model.layers[0].kernel_regularizer._get_sparse_fraction(model.layers[0]))
            fi.extend(sparsity_dynamics_step(model, x_train, y_train, l1=l1, epochs=1, remove_low_values=False))
            f.append(fi)
            s.append(model.evaluate(x_test, y_test, verbose=1))

            kernel = model.layers[0].get_weights()[0]
            with open('sparsity-nonlin-tripp-{}-{}.pkl'.format(l1, i), 'wb') as file:
                pickle.dump(kernel, file)

        fractions.append(f)
        scores.append(s)

    return fractions, scores

def sparsity_dynamics_experiment(model, x_train, y_train):
    l1s = [.0001, .001, .01, .1, 1]

    fractions = []
    for l1 in l1s:
        f = []
        for i in range(3):
            f.append(sparsity_dynamics_run(model, x_train, y_train, l1=l1))
        fractions.append(f)

    return fractions


def sparsity_dynamics_run(model, x_train, y_train, l1=.01):
    """
    :param model: freshly initialized model
    :return: fraction kernel weights non-zero every 200 minibatches
    """
    fractions = []
    fractions.append(model.layers[0].kernel_regularizer._get_sparse_fraction(model.layers[0]))
    fractions.extend(sparsity_dynamics_step(model, x_train, y_train, l1=l1))
    fractions.extend(sparsity_dynamics_step(model, x_train, y_train, l1=0.0))

    return fractions


def sparsity_control_step(model, x_train, y_train, target=.5, threshold=1e-4, epochs=1):
    batch_size = 32
    chunk_size = batch_size * 200
    n_chunks = int(np.floor(x_train.shape[0] / chunk_size))

    cycles = int(np.round(epochs*n_chunks))

    l1 = model.layers[0].kernel_regularizer.control(model.layers[0], target, threshold=threshold)
    print('regularization weight: {}'.format(l1))
    fractions = []
    for i in range(cycles):

        c = i % n_chunks
        first, last = chunk_size*c, chunk_size*(c+1)
        model.fit_generator(datagen.flow(x_train[first:last, :, :, :], y_train[first:last, :],
                                         batch_size=batch_size),
                            epochs=1,
                            workers=4)

        fractions.append(model.layers[0].kernel_regularizer._get_sparse_fraction(model.layers[0], threshold=threshold))

    return fractions, l1


def sparsity_dynamics_step(model, x_train, y_train, l1=.01, threshold=1e-4, epochs=2.85, remove_low_values=False):
    batch_size = 32
    chunk_size = batch_size * 200
    n_chunks = int(np.floor(x_train.shape[0] / chunk_size))

    cycles = int(np.round(epochs*n_chunks))

    model.layers[0].kernel_regularizer.update(new_l1=l1)
    fractions = []
    for i in range(cycles):

        if remove_low_values:
            print('removing small weights')
            weights, biases = model.layers[0].get_weights()
            weights[np.abs(weights) < threshold] = 0
            model.layers[0].set_weights((weights, biases))

        c = i % n_chunks
        first, last = chunk_size*c, chunk_size*(c+1)
        model.fit_generator(datagen.flow(x_train[first:last, :, :, :], y_train[first:last, :],
                                         batch_size=batch_size),
                            epochs=1,
                            workers=4)

        fractions.append(model.layers[0].kernel_regularizer._get_sparse_fraction(model.layers[0], threshold=threshold))

    return fractions


if __name__ == '__main__':
    batch_size = 32
    num_classes = 10
    epochs = 100
    num_predictions = 20
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_cifar10_trained_model.h5'

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = get_model()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

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

    datagen.fit(x_train)

    checkpoint_callback = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False,
                                                          save_weights_only=False, mode='auto', period=1)
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0,
                                                            mode='auto')


    fractions, l1s = control_experiment(model, x_train, y_train)
    print(fractions)
    print(l1s)

    # fractions = sparsity_dynamics_experiment(model, x_train, y_train)
    # fractions, scores = sparsity_rebound_experiment(model, x_train, y_train, x_test, y_test)

    # fractions, scores = sparsity_nonlinearity_experiment(x_train, y_train, x_test, y_test)
    # print(fractions)
    # print(scores)

    # x = np.linspace(-.1, .1, 100)
    # y = []
    # for i in range(len(x)):
    #     print(i)
    #     xi = K.variable(x[i])
    #     y.append(model.layers[0].kernel_regularizer(xi).eval(session=K.get_session()))
    #
    # import matplotlib.pyplot as plt
    # plt.plot(x, y)
    # plt.show()

    # # model.layers[0].kernel_regularizer.adjust(model.layers[0], target=.1, threshold=1e-4)
    # model.layers[0].kernel_regularizer.update(new_l1=0.0)
    # print(model.layers[0].kernel_regularizer.get_config())
    # for i in range(epochs):
    #     n = batch_size * 200
    #     print(x_train.shape)
    #     print(y_train.shape)
    #     model.fit_generator(datagen.flow(x_train[:n,:,:,:], y_train[:n,:],
    #                                      batch_size=batch_size),
    #                         epochs=1,
    #                         validation_data=(x_test, y_test),
    #                         workers=4,
    #                         callbacks=[checkpoint_callback, early_stopping_callback])
    #     # show_weights(model)
    #     print(model.layers[0].kernel_regularizer._get_sparse_fraction(model.layers[0]))
    #     model.layers[0].kernel_regularizer.update(new_l1=0.01)
    #
    # # Save model and weights
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    # model_path = os.path.join(save_dir, model_name)
    # model.save(model_path)
    # print('Saved trained model at %s ' % model_path)
    #
    # # Score trained model.
    # scores = model.evaluate(x_test, y_test, verbose=1)
    # print('Test loss:', scores[0])
    # print('Test accuracy:', scores[1])
