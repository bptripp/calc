import os
import inspect
import pickle
import keras
from keras import applications
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from calc.system import System, InterAreaProjection
from calc.network import Network
from calc.examples.example_systems import make_big_system
from calc.analysis.custom_layers import Scale
from calc.analysis.densenet121 import DenseNet
from calc.data import FV91_hierarchy

"""
For visualizing architectures of cortical models and standard convnets.   
"""

def plot_FLNe(system, figsize=(8,6), network=None):
    layers = []
    for pop in system.populations:
        layer = pop.name.split('-')[0]
        # if layer == 'Conv2D':
        #     layer = ''
        if '_2/3' in layer and not 'interblob' in layer or '_5' in layer or '_6' in layer:
            layer = ''
        if layer == 'parvo_LGN' or layer == 'konio_LGN' or layer == 'V1_4Cbeta':
            layer = ''
        layers.append(layer)

    FLNe = np.zeros((len(layers), len(layers)))
    # for projection in system.projections:

    if network is not None:
        assert len(network.connections) == len(system.projections), \
            '{} vs {}'.format(len(network.connections), len(system.projections))

    for index in range(len(system.projections)):
        projection = system.projections[index]
        i = system.find_population_index(projection.origin.name)
        j = system.find_population_index(projection.termination.name)
        if isinstance(projection, InterAreaProjection):
            # print('f: {}'.format(projection.f))
            FLNe[i,j] = projection.f
        else:
            # Make proportional to presynaptic population size (this is fairly sensible for current inter-laminar
            # structure, i.e. L4->L2/3, L4->L5, L2/3->L5, but may not be in general.
            if network is None:
                FLNe[i,j] = projection.origin.n
            else:
                print(network.connections[index].c)
                FLNe[i, j] = projection.origin.n * network.connections[index].c

    # figsize = [2*f for f in figsize]
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(np.ones((FLNe.shape[0], FLNe.shape[1], 4))) # this makes background white in .eps
    im = ax.imshow(np.log10(FLNe), vmin=np.log10(.000001), vmax=0)
    ax.set_xticks(np.arange(len(layers)))
    ax.set_yticks(np.arange(len(layers)))
    ax.set_xticklabels(layers, fontsize=7)
    ax.set_yticklabels(layers, fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
             rotation_mode="anchor")

    cbar = ax.figure.colorbar(im, ax=ax, shrink=.5)
    cbar.ax.set_ylabel(r'log$_{10}$(FLNe)', rotation=-90, va="bottom", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    # ax.set_rasterized(True)
    # plt.savefig('visualization-FLNe.pdf')
    plt.savefig('visualization-FLNe.eps')
    # plt.show()


def plot_population_sizes():
    network_names = ['MSH', 'VGG-16', 'ResNet50', 'InceptionV3', 'DenseNet121']
    symbols = ['ks', 'ro', 'b^', 'cx', 'm.']
    fig, ax = plt.subplots(figsize=(9,6))
    for i in range(len(network_names)):
        network = get_network(network_names[i])
        population_sizes = [layer.m * layer.width**2 for layer in network.layers]
        layer_names = [layer.name.split('-')[0] for layer in network.layers]
        if network_names[i] == 'MSH':
            depths = get_depths(layer_names)
            plt.plot(depths, population_sizes, symbols[i])
        else:
            plt.plot(population_sizes, symbols[i])

    plt.xlabel('Network layer')
    plt.ylabel('Number of units')
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)

    plt.yscale('log')
    plt.ylim([500, 1000000000])
    plt.legend(network_names, fontsize=15, loc='upper right')
    # plt.title('Population Sizes')
    plt.savefig('figures/population-sizes.eps')
    plt.savefig('figures/population-sizes.pdf')
    # plt.show()


def get_depths(layer_names):
    result = []
    layer_levels = {'4': 1, '4Calpha': 1, '4Cbeta': 1, '4B': 2, '2/3': 2, '2/3blob': 2, '2/3interblob': 2, '5': 3, '6': 4}
    for name in layer_names:
        if '_' in name:
            area = name.split('_')[0]
            if area.startswith('V2'):
                area = 'V2'
            layer = name.split('_')[1]
            if area in FV91_hierarchy.keys():
                cortical_level = FV91_hierarchy[area]
                layer_level = layer_levels[layer]
                depth = 1 + 4*(cortical_level-1) + layer_level
            else:
                depth = 1
        else:
            depth = 0

        result.append(depth)
        # print('{} depth: {}'.format(name, depth))

    return result


def plot_feature_maps():
    network_names = ['MSH', 'VGG-16', 'ResNet50', 'InceptionV3', 'DenseNet121']
    symbols = ['ks', 'ro', 'b^', 'cx', 'm.']
    fig, ax = plt.subplots(figsize=(9,6))
    for i in range(len(network_names)):
        network = get_network(network_names[i])
        m = [layer.m for layer in network.layers]
        # plt.plot(m, symbols[i])

        if network_names[i] == 'MSH':
            layer_names = [layer.name.split('-')[0] for layer in network.layers]
            depths = get_depths(layer_names)
            plt.plot(depths, m, symbols[i])
        else:
            plt.plot(m, symbols[i])


    #TODO: this should really be a range for macaque
    plt.xlabel('Network layer')
    plt.ylabel('Number of feature maps')
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)

    plt.yscale('log')
    plt.ylim([.5, 15000])
    plt.legend(network_names, fontsize=15, loc='lower right')
    plt.savefig('figures/feature-maps.eps')
    plt.savefig('figures/feature-maps.pdf')
    # plt.show()


def make_system(cnn):
    system = System()

    for layer in cnn.layers:
        if not _skip(layer):
            name = _get_name(layer)
            print(name)
            size = _get_size(layer)
            system.add(name, size, 1, 1)

            for input in _get_inputs(layer):
                f = _get_size(input) # this is normalized in prune_FLNe
                system.connect_areas(_get_name(input), name, f)

    system.normalize_FLNe()
    return system


def make_network(cnn):
    network = Network()

    for layer in cnn.layers:
        if not _skip(layer):
            name = _get_name(layer)
            if len(layer.output.shape) == 4:
                m = layer.output_shape[3]
                width = (layer.output_shape[1] + layer.output_shape[2]) / 2
            elif len(layer.output_shape) == 2: # dense layer
                m = layer.output_shape[1]
                width = 1
            else:
                print('No m or width for type {} with shape {}'.format(type(layer), layer.output_shape))
                m = None
                width = None

            network.add(name, m, width)

            # All networks seem to have a single w for all inputs to a given layer. For examples, Inception only has
            # convergence onto DepthConcat layers. We omit such layers as they don't have units, but they project to
            # layers that have single kernel sizes (no additional convergence).
            # To model convergence in CNN models of cortex we use linear layers and add their outputs rather than
            # concatenating.

            w, s = None, None
            if isinstance(layer, keras.layers.core.Dense):
                inputs = _get_inputs(layer)
                if len(inputs) > 1:
                    raise ValueError('Not supported')
                input = inputs[0]
                if len(input.output_shape) == 2:
                    w = 1
                elif len(input.output_shape) == 4:
                    w = (input.output_shape[1] + input.output_shape[2]) / 2
                else:
                    raise ValueError('Not supported')
            elif dir(layer).__contains__('kernel') and layer.kernel:
                w = (layer.kernel.shape[0].value + layer.kernel.shape[1].value) / 2
            elif dir(layer).__contains__('pool_size') and layer.pool_size:
                w = (layer.pool_size[0] + layer.pool_size[1]) / 2
            elif isinstance(layer, keras.layers.pooling.GlobalAveragePooling2D):
                w = (layer.input_shape[1] + layer.input_shape[2]) / 2
            else:
                print('No kernel for {}'.format(type(layer)))

            for input in _get_inputs(layer):
                if dir(input).__contains__('strides'):
                    s = (input.strides[0] + input.strides[1]) / 2
                elif isinstance(input, keras.layers.InputLayer) \
                        or isinstance(input, keras.layers.core.Dense) \
                        or isinstance(input, keras.layers.pooling.GlobalAveragePooling2D):
                    s = 1
                else:
                    print('No strides for input type: {}'.format(type(input)))

                network.connect(_get_name(input), name, 1, s, w, 1)

    return network


def _skip(layer):
    skip = False

    if isinstance(layer, keras.layers.Flatten):
        skip = True
    if isinstance(layer, keras.layers.BatchNormalization):
        skip = True
    if isinstance(layer, keras.layers.Activation):
        skip = True
    if isinstance(layer, keras.layers.ZeroPadding2D):
        skip = True
    if isinstance(layer, keras.layers.Add):
        skip = True
    if isinstance(layer, keras.layers.Concatenate):
        skip = True
    if isinstance(layer, keras.layers.Merge):
        skip = True
    if '{}'.format(type(layer)).__contains__('Scale'):
        skip = True
    # else:
    #     print(type(layer))

    return skip


def _get_size(layer):
    return np.prod(layer.output_shape[1:])


def _get_name(layer):
    return '{}-{}'.format(type(layer).__name__, id(layer))


def _get_inputs(layer):
    result = []
    for inbound_node in layer._inbound_nodes:
        for inbound_layer in inbound_node.inbound_layers:
            if _skip(inbound_layer):
                result.extend(_get_inputs(inbound_layer))
            else:
                result.append(inbound_layer)
    return result


def plot_kernel_sizes():
    macaque_net = get_network('MSH')
    max_w = max([int(np.round(conn.w)) for conn in macaque_net.connections])
    # bin_edges = np.linspace(.5, max_w+.5, max_w+1)
    bin_edges = np.linspace(.5, 70.5, 71)

    network_names = ['MSH', 'VGG-16', 'ResNet50', 'InceptionV3', 'DenseNet121']

    fig, axes = plt.subplots(nrows=len(network_names), ncols=1, figsize=(5,8))
    for i in range(len(network_names)):
        net = get_network(network_names[i])
        # w = [int(np.round(conn.w)) for conn in net.connections]
        w = [np.clip(conn.w, 0, 70) for conn in net.connections]

        if i == 0:
            actual_w = [conn.w for conn in net.connections]
            print('max width: {} n > 300: {}'.format(np.max(actual_w), sum(np.array(actual_w) > 300)))

        axes[i].hist(w, bins=bin_edges)
        axes[i].set_title(network_names[i], fontsize=18)

        for item in ([axes[i].xaxis.label, axes[i].yaxis.label] +
                     axes[i].get_xticklabels() + axes[i].get_yticklabels()):
            item.set_fontsize(16)

    xticks = [0, 10, 20, 30, 40, 50, 60, 70]
    xlabels = ['0', '10', '20', '30', '40', '50', '60', '70+']
    for axis in axes:
        axis.set_xticks(xticks)
        axis.set_xticklabels(xlabels)

    axes[2].set_ylabel('Count')
    axes[-1].set_xlabel('Kernel width')
    plt.tight_layout()
    plt.savefig('figures/kernel-width.eps')
    plt.savefig('figures/kernel-width.pdf')
    plt.show()


def plot_stride():
    macaque_net = get_network('MSH')
    max_s = max([int(np.round(conn.s)) for conn in macaque_net.connections])
    bin_edges = np.linspace(.5, max_s+.5, max_s+1)

    network_names = ['MSH', 'VGG-16', 'ResNet50', 'InceptionV3', 'DenseNet121']
    # network_names = ['Macaque', 'VGG-16']

    fig, axes = plt.subplots(nrows=len(network_names), ncols=1, figsize=(5,8))
    for i in range(len(network_names)):
        net = get_network(network_names[i])
        s = [int(np.round(conn.s)) for conn in net.connections]
        n, bins, patches = axes[i].hist(s, bins=bin_edges)
        axes[i].set_title(network_names[i], fontsize=18)
        print(n)

        for j in range(len(n)):
            if n[j] > 0 and n[j] < .05*np.max(n):
                axes[i].text((bins[j] + bins[j+1])/2 - .2, np.max(n)/10, '*', fontsize=16)

        for item in ([axes[i].xaxis.label, axes[i].yaxis.label] +
                     axes[i].get_xticklabels() + axes[i].get_yticklabels()):
            item.set_fontsize(16)


    axes[2].set_ylabel('Count')
    axes[-1].set_xlabel('Stride')
    plt.tight_layout()
    plt.savefig('figures/stride.eps')
    plt.savefig('figures/stride.pdf')
    plt.show()


def get_skip_lengths(system):
    """
    :param system: a system
    :return: length of skip connection for each projection in the system, i.e. the length
        of the longest path between the nodes
    """
    import networkx as nx

    g = system.make_graph()
    for u, v, d in g.edges(data=True):
        d['weight'] = -1

    lengths = {}
    for pop in system.populations:
        lengths[pop.name] = nx.bellman_ford(g, pop.name)

    result = []
    for p in system.projections:
        result.append(-lengths[p.origin.name][1][p.termination.name])

    # print some additional information ...
    print('longest skip connection: {}'.format(max(result)))
    for i in range(len(system.projections)):
        if result[i] == max(result):
            print([system.projections[i].origin.name, system.projections[i].termination.name])
    print('longest path: {}'.format(nx.dag_longest_path_length(g)))
    print(nx.dag_longest_path(g))

    return result


def plot_sparsity():
    macaque_net = get_network('MSH')
    sparsity = [conn.c * conn.sigma for conn in macaque_net.connections]
    # sparsity = [conn.c for conn in macaque_net.connections]
    kernel_width = [conn.w for conn in macaque_net.connections]

    macaque_system = get_system('Macaque')
    inter_area = [isinstance(p, InterAreaProjection) for p in macaque_system.projections]

    skip_lengths = get_skip_lengths(macaque_system)

    inter_area_sparsity = []
    inter_area_kernel_width = []
    inter_area_skip_lengths = []
    inter_laminar_sparsity = []
    inter_laminar_kernel_width = []
    inter_laminar_skip_lengths = []
    for i in range(len(inter_area)):
        if inter_area[i]:
            inter_area_sparsity.append(sparsity[i])
            inter_area_kernel_width.append(kernel_width[i])
            inter_area_skip_lengths.append(skip_lengths[i])
        else:
            inter_laminar_sparsity.append(sparsity[i])
            inter_laminar_kernel_width.append(kernel_width[i])
            inter_laminar_skip_lengths.append(skip_lengths[i])


    plt.figure(figsize=(3.5,2.5))
    plt.loglog(inter_area_kernel_width, inter_area_sparsity, 'kx')
    plt.loglog(inter_laminar_kernel_width, inter_laminar_sparsity, 'b.')
    plt.xlabel('kernel width')
    plt.ylabel('sparsity')
    # plt.legend(('inter-area', 'inter-laminar'), loc='upper right')
    plt.tight_layout()
    plt.savefig('figures/kernel-vs-sparsity.eps')
    plt.show()

    plt.figure(figsize=(3.5,2.5))
    plt.semilogy(inter_area_skip_lengths, inter_area_kernel_width, 'kx')
    plt.semilogy(inter_laminar_skip_lengths, inter_laminar_kernel_width, 'b.')
    plt.xlabel('skip length')
    plt.ylabel('kernel width')
    plt.legend(('inter-area', 'inter-laminar'), loc='lower right')
    plt.tight_layout()
    plt.savefig('figures/skip-length-vs-kernel.eps')
    plt.show()


def get_system(name):
    if name.lower() == 'macaque':
        system = make_big_system()
        graph = system.make_graph()

        def get_in_degree(area):
            result = graph.in_degree('{}_4'.format(area))
            if result == {}:
                raise RuntimeError(area)
            else:
                return result

        name_order = ['INPUT', 'parvo_LGN', 'magno_LGN', 'konio_LGN',
                'V1_4Calpha', 'V1_4Cbeta', 'V1_4B',
                'V1_2/3blob', 'V1_2/3interblob', 'V1_5', 'V1_6',
                'V2thick_4', 'V2thick_2/3', 'V2thick_5', 'V2thick_6',
                'V2thin_4', 'V2thin_2/3', 'V2thin_5', 'V2thin_6',
                'V2pale_4', 'V2pale_2/3', 'V2pale_5', 'V2pale_6']

        exclusions = ['MDP', '7b', '36']
        for level in range(3, 11):
            areas_this_level = []
            for key in FV91_hierarchy.keys():
                if FV91_hierarchy[key] == level and not key in exclusions:
                    areas_this_level.append(key)

            areas_this_level = sorted(areas_this_level, key=get_in_degree)
            for area in areas_this_level:
                for layer in ['4', '2/3', '5', '6']:
                    name_order.append('{}_{}'.format(area, layer))

        populations = [system.find_population(name) for name in name_order]

        print([p.name for p in system.populations])
        assert len(system.populations) == len(populations), '{} {}'.format(len(system.populations), len(populations))
        system.populations = populations

        return system
    else:
        cnn = load_cnn(name)
        return make_system(cnn)


def get_network(name):
    if name.lower() == 'macaque' or name == 'MSH': # load saved optimized network
        # filename = '../optimization-result-fixed-f.pkl'
        filename = 'optimization-result-msh-best.pkl'
        # filename = 'optimization-result-msh-best-2.pkl'
        # filename = 'optimization-result-msh-0.pkl'
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return data['net']
    else:
        cnn = load_cnn(name)
        return make_network(cnn)


def count_parameters(net):
    sparse_result = 0
    dense_result = 0
    for c in net.connections:
        full_size = c.pre.m * c.post.m * c.w**2
        sparsity_fraction = c.sigma * c.c
        sparse_result += full_size * sparsity_fraction
        dense_result += full_size * c.c

    print('# parameters (ignoring pooling etc.): {} (sparse) {} (dense)'.format(sparse_result, dense_result))


def fix_network():
    """
    I accidentally included V2 in the network in addition to it's stripes. It has input but no output,
    so this won't affect the optimization of other areas/connections, but I have to remove V2.
    """
    filename = 'optimization-result-msh-1.pkl'

    with open(filename, 'rb') as file:
        data = pickle.load(file)

    net = data['net']

    layers = []
    for layer in net.layers:
        if not 'V2_' in layer.name:
            layers.append(layer)

    connections = []
    for connection in net.connections:
        if not ('V2_' in connection.pre.name or 'V2_' in connection.post.name):
            connections.append(connection)

    net.layers = layers
    net.connections = connections

    data['net'] = net

    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def load_cnn(name):
    if name == 'VGG-16':
        return applications.VGG16()
    elif name == 'InceptionV3':
        return applications.InceptionV3(input_shape=(255, 255, 3))
    elif name == 'ResNet50':
        return applications.ResNet50()
    elif name == 'DenseNet121':
        # weights_path = '{}/imagenet_models/densenet121_weights_tf.h5'.format(os.path.dirname(inspect.stack()[0][1]))
        weights_path = '/Users/bptripp/code/calc-old/imagenet_models/densenet121_weights_tf.h5'.format(os.path.dirname(inspect.stack()[0][1]))
        return DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)
    else:
        raise ValueError('Unknown CNN: {}'.format(name))


if __name__ == '__main__':
    # system = get_system('InceptionV3')
    # figsize=(8,8)
    # system = get_system('ResNet50')
    # figsize=(7,7)
    system = get_system('VGG-16')
    figsize=(4,4)
    # system = get_system('DenseNet121')
    # figsize=(8,8)
    plot_FLNe(system, figsize=figsize)

    # system = get_system('macaque')
    # network = get_network('macaque')
    #
    # system.print_description()
    # network.print()

    # figsize=(10,10)
    # plot_FLNe(system, figsize=figsize, network=network)
    # print(len(system.populations))

    # plot_population_sizes()
    # plot_feature_maps()

    # network = get_network('VGG-16')
    # network = get_network('InceptionV3')
    # network = get_network('macaque')
    # network.print()
    # # network = get_network('DenseNet121')
    # get_network('VGG-16')
    # plot_kernel_sizes()
    # plot_stride()

    # macaque_system = get_system('Macaque')
    # print(get_skip_lengths(macaque_system))
    # plot_sparsity()

    # count_parameters(get_network('Macaque'))
