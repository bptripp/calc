import copy
import numpy as np

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