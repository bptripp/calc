import random
import copy
import numpy as np
import pickle
import keras
from keras.layers import Conv2D, Activation, BatchNormalization, Lambda, Dropout
from keras.constraints import Constraint
from keras.initializers import VarianceScaling
from keras.regularizers import l2
from keras import backend as K


def subsample_maps(net):
    """
    Randomly samples presynaptic feature maps that contribute to each connection.
    We try to make sets of maps of a given layer that contribute to different
    interarea connections mostly disjoint, although this doesn't have to be done
    strictly, and in any case it's not always possible as the total across connections
    may be greater than the number of maps. We do the same for interlaminar connections.

    :param net: a Network
    :return: indices of presynaptic feature maps to use in each connection
    """

    # start with uniform probability that each map is selected for inclusion in a connection;
    # reduce probability of selecting a given map again whenever it is selected
    probability_reduction = .01
    interarea_probabilities = []
    interlaminar_probabilities = []
    for layer in net.layers:
        interarea_probabilities.append(np.ones(layer.m))
        interlaminar_probabilities.append(np.ones(layer.m))

    subsample_indices = []
    for connection in net.connections:
        n_maps = int(round(connection.pre.m * connection.c))
        pre_index = net.find_layer_index(connection.pre.name)

        pre_cortical_area = connection.pre.name.split('_')[0]
        post_cortical_area = connection.post.name.split('_')[0]
        if pre_cortical_area == post_cortical_area: # interlaminar
            p = interlaminar_probabilities[pre_index]
        else:
            p = interarea_probabilities[pre_index]

        indices = np.random.choice(range(int(connection.pre.m)), n_maps, replace=False, p=p/np.sum(p))
        for i in indices:
            p[i] = p[i]*probability_reduction
        subsample_indices.append(indices)

    # for i in range(len(net.layers)):
    #     print(net.layers[i].name)
    #     print(np.mean(interarea_probabilities[i]))
    #     print(np.mean(interlaminar_probabilities[i]))

    return subsample_indices


def prune_maps(net, subsample_indices, output_name):
    """
    In case not all the feature maps are used in feedforward connections, the unused ones
    can optionally be omitted from a model, if the model is not to include feedback
    or lateral connections.

    :param net:
    :param subsample_indices: result of subsample_maps; indices of maps that provide input to each connection
    :param output_name: name of layer used to feed output (not pruned)
    :return:
    """

    # find which feature maps are used in output
    indices_by_layer = []
    for layer in net.layers:
        connections = net.find_outbounds(layer.name)

        all_indices_for_layer = []
        for connection in connections:
            conn_ind = net.find_connection_index(connection.pre.name, connection.post.name)
            all_indices_for_layer.extend(subsample_indices[conn_ind])

        all_indices_for_layer = list(set(all_indices_for_layer))
        all_indices_for_layer.sort()
        indices_by_layer.append(all_indices_for_layer)

        # print('layer {}'.format(layer.name))
        # print(all_indices_for_layer)

    new_subsample_indices = copy.deepcopy(subsample_indices)

    # discard unused maps and condense indices
    for i in range(len(net.layers)):
        layer = net.layers[i]

        if not layer.name == output_name and not layer.name == 'INPUT':
            connections = net.find_outbounds(layer.name)

            for connection in connections:
                ind = net.find_connection_index(connection.pre.name, connection.post.name)
                for j in range(len(new_subsample_indices[ind])):
                    new_subsample_indices[ind][j] = indices_by_layer[i].index(subsample_indices[ind][j])

            old_m = layer.m
            layer.m = len(indices_by_layer[i])

            # print('{}: {} layers condensed to {}'.format(layer.name, old_m, layer.m))
            # for connection in connections:
            #     ind = net.find_connection_index(connection.pre.name, connection.post.name)
            #     print('{} condensed to {}'.format(subsample_indices[ind], new_subsample_indices[ind]))
            # print('-------')

    return new_subsample_indices


def prune_connections(net, subsample_indices):
    """
    Remove connections with no subsample indices.

    :param net:
    :param subsample_indices:
    :return:
    """
    new_connections = []
    new_subsample_indices = []
    for i in range(len(subsample_indices)):
        if len(subsample_indices[i]) > 0:
            new_connections.append(net.connections[i])
            new_subsample_indices.append(subsample_indices[i])

    net.connections = new_connections
    return new_subsample_indices


def prune_layers(net, subsample_indices):
    """
    Remove layers with no maps and their associated connections. It is possible
    for a layers to lose all its maps in prune_maps().

    :param net: a Network
    :return: updated Network with empty layers and associated connections removed
    """
    new_layers = []

    discarded_names = []
    for layer in net.layers:
        if layer.m > 0:
            new_layers.append(layer)
        else:
            discarded_names.append(layer.name)

    new_connections = []
    new_subsample_indices = []
    for i in range(len(net.connections)):
        connection = net.connections[i]
        if connection.pre.name not in discarded_names and connection.post.name not in discarded_names:
            new_connections.append(connection)
            new_subsample_indices.append(subsample_indices[i])

    net.layers = new_layers
    net.connections = new_connections

    return new_subsample_indices


def get_map_list(x):
    result = []
    for i in range(x.shape[-1]):
        xi = Lambda(lambda x: x[:, :, :, i:(i+1)])(x) #need colon, otherwise a dimension is dropped
        result.append(xi)
    return result


def get_map_concatenation(map_list, indices):
    if len(indices) == 1:
        return map_list[indices[0]]
    else:
        short_list = []
        for index in indices:
            print('index: {} map_list size: {}'.format(index, len(map_list)))
            short_list.append(map_list[index])
        return keras.layers.concatenate(short_list)


def make_model_from_network(net, input, output_name, subsample_indices=None):
    """
    Note the "output" layer returned from this method may not be the model's final output,
    but rather the nearest layer to the output with a direct physiological homologue. You
    can add layers after it to suit a certain task. For example, for CIFAR-10 you might have:

    output_layer = make_model_from_network(net, input, 'TEpd_5')
    layer_f = Flatten()(output_layer)
    layer_d1 = Dense(128, activation='relu')(layer_f)
    layer_d2 = Dense(128, activation='relu')(layer_d1)
    layer_classifier = Dense(10, activation='softmax')(layer_d2)
    model = keras.Model(inputs=input, outputs=layer_classifier)

    :param input: layer that provides images
    :param output_name: name of layer to use as output
    :return: output_layer: output layer (other layers can be added after this; see above)
    """

    input_name = 'INPUT'
    # dropout_layer = Dropout(0.5)(input)
    complete_layers = {input_name: input}
    # kernel_masks = {}

    sparse_layers = []
    # print(len(complete_layers))
    # print(complete_layers)

    while len(complete_layers) < len(net.layers):
        for layer in net.layers:
            # print('*****')
            # print(layer.name)
            if layer.name not in complete_layers.keys():
                # add this layer if all its inputs are there already
                inbounds = net.find_inbounds(layer.name)

                all_there = True
                for inbound in inbounds:
                    if inbound.pre.name not in complete_layers.keys():
                        all_there = False
                        break

                if all_there:
                    conv_layers = [] #one for each input
                    for inbound in inbounds:
                        inbound_index = net.find_connection_index(inbound.pre.name, inbound.post.name)

                        m = int(np.round(layer.m))
                        w = int(np.round(inbound.w))
                        s = int(np.round(inbound.s))

                        if m == 0 or w == 0:
                            print('origin: {} termination: {} m: {} w: {} stride: {}'.format(inbound.pre.name, layer.name, m, w, s))

                        name = '{}-{}'.format(inbound.pre.name, layer.name)
                        input_layer = complete_layers[inbound.pre.name]
                        print('connecting {} c: {} sigma: {}'.format(name, inbound.c, inbound.sigma))

                        # print('subsample indices: {}'.format(subsample_indices[inbound_index]))
                        if len(subsample_indices[inbound_index]) > 0:
                            ml = get_map_list(input_layer) #TODO: build a single list that's shared across connections
                            print('{} of {}'.format(inbound_index, len(subsample_indices)))
                            subsampled = get_map_concatenation(ml, subsample_indices[inbound_index])

                            kernel_constraint = SparsityConstraint(inbound.sigma)
                            # kernel_constraint.set_random_mask((w, w), m, len(subsample_indices[inbound_index]))
                            # kernel_masks[name] = kernel_constraint.non_zero
                            conv_layer = Conv2D(m, (w, w), strides=(s, s), padding='same', name=name,
                                                kernel_constraint=kernel_constraint,
                                                kernel_initializer=scaled_glorot(inbound.sigma),
                                                kernel_regularizer=l2(0.000001),
                                                )
                            conv_outputs = conv_layer(subsampled)
                            conv_layers.append(conv_outputs) #TODO: clean up names
                            sparse_layers.append(conv_layer)

                    if len(conv_layers) > 1:
                        x = keras.layers.add(conv_layers)
                    else:
                        x = conv_layers[0]

                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)
                    # x = Dropout(.032)(x)

                    complete_layers[layer.name] = x

                    print("adding " + layer.name)

    # with open('kernel_masks.pkl', 'wb') as file:
    #     pickle.dump(kernel_masks, file)

    return complete_layers[output_name], sparse_layers


def snip(sparse_layers, model, inputs, outputs):
    weights = [layer.kernel for layer in sparse_layers]

    # gradients = keras.backend.gradients(loss, variables)
    # print(gradients)
    # weights = model.trainable_weights
    # input_tensors = model.inputs + model.sample_weights + model.targets + [K.learning_phase()]

    gradients = model.optimizer.get_gradients(model.total_loss, weights)
    # input_tensors = model.inputs + model.targets
    # get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    # get_weights = K.function(inputs=input_tensors, outputs=weights)

    # g = get_gradients(batch) #TODO
    # w = get_weights(batch)

    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)

    f = K.function(symb_inputs, gradients)
    g = f(x + y + sample_weight)

    w = [layer.get_weights()[0] for layer in sparse_layers]
    for i in range(len(w)):
        sparse_layers[i].kernel_constraint.set_snip_mask(w[i], g[i])


def scaled_glorot(sigma):
    """
    :param sigma: pixel-wise kernel sparseness parameter
    :return: Keras glorot_uniform initializer scaled as 1/sigma
    """
    return VarianceScaling(scale=1./sigma**.5,
                           mode='fan_avg',
                           distribution='uniform',
                           seed=None)


def test_subsample_of_non_continuous_maps():
    from keras.layers import Input, Lambda, Concatenate
    x = Input((28, 28, 1))
    x = Conv2D(3, (5, 5))(x)

    # x0 = Lambda(lambda x: x[:, :, :, 0:1])(x) #need colon, otherwise a dimension is dropped
    # x2 = Lambda(lambda x: x[:, :, :, 2:3])(x)
    # x02 = keras.layers.concatenate([x0, x2])

    ml = get_map_list(x)
    x0 = ml[0]
    x02 = get_map_concatenation(ml, [0, 2])

    print(x.shape)
    print(x0.shape)
    print(x02.shape)


class SparsityConstraint(Constraint):
    """
    Maintains kernel sparseness by resetting some entries to 0 at each step.

    :param sigma: weight-level sparsity parameter (fraction non-zero elements)
    """

    def __init__(self, sigma):
        self.non_zero = None
        self.mask = None
        self.sigma = sigma
        # self.description = 'kernel: {} pre_maps: {} maps: {}'.format(kernel_shape, pre_maps, post_maps)

    def set_random_mask(self, kernel_shape, post_maps, pre_maps):
        """
        :param kernel_shape: kernel shape (width, height)
        :param post_maps: number of postsynaptic feature maps
        :param pre_maps: number of presynaptic feature maps
        :param sigma: weight-level sparsity parameter (fraction non-zero elements)
        """
        self.non_zero = (np.random.rand(kernel_shape[0], kernel_shape[1], pre_maps, post_maps) <= self.sigma)
        self.mask = K.constant(1 * self.non_zero)

    def set_snip_mask(self, weights, gradients):
        """
        Sets mask according to the SNIP sparsification method (Lee et al., 2019, ICLR).

        :param weights: kernel weights
        :param gradients: gradients of loss (sampled from some batch) wrt weights
        """
        absolute_sensitivities = np.abs(weights * gradients)
        n_to_keep = int(absolute_sensitivities.size * self.sigma)

        threshold = np.sort(absolute_sensitivities.flatten())[-n_to_keep]
        self.mask = K.constant(absolute_sensitivities >= threshold)

    def __call__(self, w):
        # print(self.description)
        # print(w.shape)
        # print(self.mask.shape)
        return self.mask * w


if __name__ == '__main__':
    # import pickle
    # with open('../generated-files/calc-training-ventral-mini-2.pickle', 'rb') as f:
    #     data = pickle.load(f)
    # net = data['nets'][4] #smallest (Trainable params: 92,336,653)
    #
    # subsample_indices = subsample_maps(net)
    # subsample_indices = prune_maps(net, subsample_indices, 'TEpd_5')

    test_subsample_of_non_continuous_maps()