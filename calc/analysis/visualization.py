import keras
from keras import applications
import numpy as np
import matplotlib.pyplot as plt
from calc.system import System, InterAreaProjection
from calc.examples.example_systems import make_big_system

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
            FLNe[i,j] = projection.f
        else:
            FLNe[i,j] = 1

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
                system.connect_areas(_get_name(input), name, _get_size(input))

    system.prune_FLNe()
    return system


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


if __name__ == '__main__':
    # cnn = applications.InceptionV3(input_shape=(255,255,3))
    # figsize=(8,8)

    # cnn = applications.ResNet50()
    # figsize=(7,7)

    # cnn = applications.VGG16()
    # figsize=(4,4)

    # system = make_system(cnn)


    system = make_big_system()
    figsize=(8,8)

    plot_FLNe(system, figsize=figsize)
