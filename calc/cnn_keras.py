import random
import copy
import numpy as np
import keras
from keras.layers import Conv2D, Activation, BatchNormalization, Lambda, Concatenate


def subsample_maps(net):
    subsample_indices = []
    for connection in net.connections:
        n_maps = int(round(connection.pre.m * connection.c))
        subsample_indices.append(random.sample(range(int(connection.pre.m)), n_maps))
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

        if not layer.name == output_name:
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
    complete_layers = {input_name: input}

    print(len(complete_layers))
    print(complete_layers)

    while len(complete_layers) < len(net.layers):
        for layer in net.layers:
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

                        m = int(layer.m)
                        w = int(inbound.w)
                        s = int(inbound.s)
                        print('origin: {} termination: {} m: {} w: {} stride: {}'.format(inbound.pre.name, layer.name, m, w, s))
                        name = '{}-{}'.format(inbound.pre.name, layer.name)
                        input_layer = complete_layers[inbound.pre.name]

                        if subsample_indices[inbound_index]:
                            ml = get_map_list(input_layer) #TODO: build a single list that's shared across connections
                            subsampled = get_map_concatenation(ml, subsample_indices[inbound_index])

                            # conv_layer = Conv2D(m, (w, w), strides=(s, s), padding='same', name=name)(input_layer)
                            conv_layer = Conv2D(m, (w, w), strides=(s, s), padding='same', name=name)(subsampled)
                            conv_layers.append(conv_layer)

                    if len(conv_layers) > 1:
                        print('adding converging paths')
                        x = keras.layers.add(conv_layers)
                    else:
                        x = conv_layers[0]

                    x = Activation('relu')(x)
                    x = BatchNormalization()(x)
                    complete_layers[layer.name] = x

                    print("adding " + layer.name)

    return complete_layers[output_name]

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


if __name__ == '__main__':
    # import pickle
    # with open('../generated-files/calc-training-ventral-mini-2.pickle', 'rb') as f:
    #     data = pickle.load(f)
    # net = data['nets'][4] #smallest (Trainable params: 92,336,653)
    #
    # subsample_indices = subsample_maps(net)
    # subsample_indices = prune_maps(net, subsample_indices, 'TEpd_5')

    test_subsample_of_non_continuous_maps()