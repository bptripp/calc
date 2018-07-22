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

"""
For visualizing architectures of cortical models and standard convnets.   
"""


def plot_FLNe(system, figsize=(8,6)):
    layers = []
    for pop in system.populations:
        layers.append(pop.name.split('-')[0])

    FLNe = np.zeros((len(layers), len(layers)))
    for projection in system.projections:
        i = system.find_population_index(projection.origin.name)
        j = system.find_population_index(projection.termination.name)
        if isinstance(projection, InterAreaProjection):
            # print('f: {}'.format(projection.f))
            FLNe[i,j] = projection.f
        else:
            # Make proportional to presynaptic population size (this is fairly sensible for current inter-laminar
            # structure, i.e. L4->L2/3, L4->L5, L2/3->L5, but may not be in general.
            FLNe[i,j] = projection.origin.n

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
    network_names = ['Macaque', 'VGG-16', 'ResNet50', 'InceptionV3', 'DenseNet121']
    symbols = ['ks', 'ro', 'b^', 'cx', 'm.']
    fig, ax = plt.subplots(figsize=(9,6))
    for i in range(len(network_names)):
        network = get_network(network_names[i])
        population_sizes = [layer.m * layer.width**2 for layer in network.layers]
        layer_names = [layer.name.split('-')[0] for layer in network.layers]
        plt.plot(population_sizes, symbols[i])

    plt.xlabel('Network layer')
    plt.ylabel('Number of units')
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)

    plt.yscale('log')
    plt.ylim([500, 1000000000])
    plt.legend(network_names, fontsize=15, loc='upper right')
    plt.savefig('population-sizes.eps')
    plt.savefig('population-sizes.pdf')
    plt.show()


def plot_feature_maps():
    network_names = ['Macaque', 'VGG-16', 'ResNet50', 'InceptionV3', 'DenseNet121']
    symbols = ['ks', 'ro', 'b^', 'cx', 'm.']
    fig, ax = plt.subplots(figsize=(9,6))
    for i in range(len(network_names)):
        network = get_network(network_names[i])
        m = [layer.m for layer in network.layers]
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
    plt.savefig('feature-maps.eps')
    plt.savefig('feature-maps.pdf')
    plt.show()


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
    macaque_net = get_network('Macaque')
    max_w = max([int(np.round(conn.w)) for conn in macaque_net.connections])
    bin_edges = np.linspace(.5, max_w+.5, max_w+1)

    network_names = ['Macaque', 'VGG-16', 'ResNet50', 'InceptionV3', 'DenseNet121']

    fig, axes = plt.subplots(nrows=len(network_names), ncols=1, figsize=(5,8))
    for i in range(len(network_names)):
        net = get_network(network_names[i])
        w = [int(np.round(conn.w)) for conn in net.connections]
        axes[i].hist(w, bins=bin_edges)
        axes[i].set_title(network_names[i], fontsize=18)

        for item in ([axes[i].xaxis.label, axes[i].yaxis.label] +
                     axes[i].get_xticklabels() + axes[i].get_yticklabels()):
            item.set_fontsize(16)

    axes[2].set_ylabel('Count')
    axes[-1].set_xlabel('Kernel width')
    plt.tight_layout()
    plt.savefig('kernel-width.eps')
    plt.savefig('kernel-width.pdf')
    plt.show()


def plot_stride():
    macaque_net = get_network('Macaque')
    max_s = max([int(np.round(conn.s)) for conn in macaque_net.connections])
    bin_edges = np.linspace(.5, max_s+.5, max_s+1)

    network_names = ['Macaque', 'VGG-16', 'ResNet50', 'InceptionV3', 'DenseNet121']
    # network_names = ['Macaque', 'VGG-16']

    fig, axes = plt.subplots(nrows=len(network_names), ncols=1, figsize=(5,8))
    for i in range(len(network_names)):
        net = get_network(network_names[i])
        s = [int(np.round(conn.s)) for conn in net.connections]
        axes[i].hist(s, bins=bin_edges)
        axes[i].set_title(network_names[i], fontsize=18)

        for item in ([axes[i].xaxis.label, axes[i].yaxis.label] +
                     axes[i].get_xticklabels() + axes[i].get_yticklabels()):
            item.set_fontsize(16)

    axes[2].set_ylabel('Count')
    axes[-1].set_xlabel('Stride')
    plt.tight_layout()
    plt.savefig('stride.eps')
    plt.savefig('stride.pdf')
    plt.show()


def get_system(name):
    if name == 'macaque':
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
    if name.lower() == 'macaque': # load saved optimized network
        with open('../optimization-result-best-of-500-2.pkl', 'rb') as file:
            data = pickle.load(file)
        return data['net']
    else:
        cnn = load_cnn(name)
        return make_network(cnn)


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
    system = get_system('DenseNet121')
    figsize=(10,10)
    # system = get_system('macaque')
    # figsize=(10,10)
    plot_FLNe(system, figsize=figsize)

    # plot_population_sizes()
    # plot_feature_maps()

    # network = get_network('InceptionV3')
    # network = get_network('macaque')
    # network.print()
    # # network = get_network('DenseNet121')
    # get_network('VGG-16')
    # plot_kernel_sizes()
    # plot_stride()



