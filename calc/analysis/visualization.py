import os
import inspect
import pickle
import keras
from keras import applications
import numpy as np
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
        if layer == 'Conv2D':
            layer = ''
        if '_2/3' in layer or '_5' in layer:
            layer = ''
        layers.append(layer)

    FLNe = np.zeros((len(layers), len(layers)))
    # for projection in system.projections:
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


    fig, ax = plt.subplots(figsize=figsize)
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
    plt.savefig('visualization-FLNe.pdf')
    plt.show()


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
    layer_levels = {'4': 1, '4Calpha': 1, '4Cbeta': 1, '4B': 1, '2/3': 2, '5': 3}
    for name in layer_names:
        if '_' in name:
            area = name.split('_')[0]
            if area.startswith('V2'):
                area = 'V2'
            layer = name.split('_')[1]
            if area in FV91_hierarchy.keys():
                cortical_level = FV91_hierarchy[area]
                layer_level = layer_levels[layer]
                depth = 1 + 3*(cortical_level-1) + layer_level
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

    system.prune_FLNe()
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
    plt.savefig('kernel-width.eps')
    plt.savefig('kernel-width.pdf')
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
    plt.savefig('stride.eps')
    plt.savefig('stride.pdf')
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

        name_order = ['INPUT', 'parvo_LGN', 'magno_LGN', 'konio_LGN', 'V1_4Calpha', 'V1_4Cbeta', 'V1_4B', 'V1_2/3', 'V1_5',
         'V2thick_2/3', 'V2thick_4', 'V2thick_5', 'V2thin_2/3', 'V2thin_4', 'V2thin_5', 'VP_2/3', 'VP_4', 'VP_5', 'V3_2/3',
         'V3_4', 'V3_5', 'V3A_2/3', 'V3A_4', 'V3A_5', 'PIP_2/3', 'PIP_4', 'PIP_5', 'MT_2/3', 'MT_4', 'MT_5', 'V4t_2/3', 'V4t_4', 'V4t_5', 'V4_2/3',
         'V4_4', 'V4_5', 'VOT_2/3', 'VOT_4', 'VOT_5', 'PO_2/3',
         'PO_4', 'PO_5', 'DP_2/3', 'DP_4', 'DP_5', 'MSTd_2/3', 'MSTd_4', 'MSTd_5', 'MIP_2/3', 'MIP_4', 'MIP_5', 'VIP_2/3', 'VIP_4', 'VIP_5', 'LIP_2/3',
         'LIP_4', 'LIP_5', 'PITv_2/3', 'PITv_4', 'PITv_5', 'PITd_2/3', 'PITd_4', 'PITd_5', 'MSTl_2/3', 'MSTl_4', 'MSTl_5',
         'CITv_2/3', 'CITv_4', 'CITv_5', 'CITd_2/3', 'CITd_4', 'CITd_5', 'AITv_2/3',
         'AITv_4', 'AITv_5', 'FST_2/3', 'FST_4', 'FST_5', 'FEF_2/3', 'FEF_4', 'FEF_5', '7a_2/3', '7a_4', '7a_5', 'STPp_2/3',
         'STPp_4', 'STPp_5', 'STPa_2/3', 'STPa_4', 'STPa_5', 'AITd_2/3', 'AITd_4', 'AITd_5', '46_2/3', '46_4', '46_5',
         'TF_2/3', 'TF_4', 'TF_5', 'TH_2/3', 'TH_4', 'TH_5']
        populations = [system.find_population(name) for name in name_order]
        system.populations = populations

        return system
    else:
        cnn = load_cnn(name)
        return make_system(cnn)


def get_network(name):
    if name.lower() == 'macaque' or name == 'MSH': # load saved optimized network
        with open('../optimization-result-fixed-f.pkl', 'rb') as file:
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


def load_cnn(name):
    if name == 'VGG-16':
        return applications.VGG16()
    elif name == 'InceptionV3':
        return applications.InceptionV3(input_shape=(255, 255, 3))
    elif name == 'ResNet50':
        return applications.ResNet50()
    elif name == 'DenseNet121':
        weights_path = '{}/imagenet_models/densenet121_weights_tf.h5'.format(os.path.dirname(inspect.stack()[0][1]))
        return DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)
    else:
        raise ValueError('Unknown CNN: {}'.format(name))


if __name__ == '__main__':
    # system = get_system('InceptionV3')
    # figsize=(8,8)
    # system = get_system('ResNet50')
    # figsize=(7,7)
    # system = get_system('VGG-16')
    # figsize=(4,4)
    # system = get_system('DenseNet121')
    # figsize=(9,9)
    # system = get_system('macaque')
    # network = get_network('macaque')
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

    macaque_system = get_system('Macaque')
    print(get_skip_lengths(macaque_system))
    # plot_sparsity()

    # count_parameters(get_network('Macaque'))
