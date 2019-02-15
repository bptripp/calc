"""
End-to-end testing of an example optimization result.

It is possible to run these tests as unit tests, however it makes more sense to run the
script, which takes several minutes, because it runs an optimization as a first step.
The optimization result is saved for convenience. However, if a bug is introduced in the system,
these tests won't see it until the optimization runs again.
"""

import pytest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from calc.system import System
from calc.optimization import optimize_network_architecture


def small_system():
    result = System()
    result.add_input(750000, .02)
    result.add('V1_4', 53000000, 500, .09)
    result.add('V1_23', 53000000, 1000, .1)
    result.add('V1_5', 27000000, 3000, .11)
    result.add('V2_4', 33000000, 500, .19)
    result.add('V2_23', 33000000, 1000, .2)
    result.add('V2_5', 17000000, 3000, .21)
    result.add('V4_4', 17000000, 500, .39)
    result.add('V4_23', 17000000, 1000, .4)
    result.add('V4_5', 8000000, 3000, .41)

    result.connect_areas('INPUT', 'V1_4', 1.)

    result.connect_layers('V1_4', 'V1_23', 800.)
    result.connect_layers('V1_23', 'V1_5', 3000.)

    result.connect_areas('V1_5', 'V2_4', 1.)

    result.connect_layers('V2_4', 'V2_23', 800.)
    result.connect_layers('V2_23', 'V2_5', 3000.)

    result.connect_areas('V1_5', 'V4_4', .15)
    result.connect_areas('V2_5', 'V4_4', .85)

    result.connect_layers('V4_4', 'V4_23', 800.)
    result.connect_layers('V4_23', 'V4_5', 3000.)

    return result


def optimize_and_save_network():
    system = small_system()
    net, training_curve = optimize_network_architecture(system, stride_pattern=None, compare=True)

    with open('test-network.pkl', 'wb') as file:
        pickle.dump(net, file)


def load_network():
    with open('test-network.pkl', 'rb') as file:
        net = pickle.load(file)
    return net


def test_stride():
    system = small_system()
    net = load_network()

    graph = system.make_graph()
    path = nx.algorithms.dag.dag_longest_path(graph)

    cumulative_stride = 1
    for i in range(1, len(path)):
        pre = path[i-1]
        post = path[i]
        ci = net.find_connection_index(pre, post)
        cumulative_stride *= net.connections[ci].s

    assert net.find_layer(path[0]).width / cumulative_stride == net.find_layer(path[-1]).width


def test_rf_size(plot=False):
    system = small_system()
    net = load_network()

    graph = system.make_graph()
    path = nx.algorithms.dag.dag_longest_path(graph)

    cumulative_stride = 1

    input_pixel_width = system.find_population(path[0]).w

    indices = np.arange(0, 121, 1)
    rf = np.zeros_like(indices)
    rf[60] = 1

    if plot: plt.figure()

    for i in range(1, len(path)):
        pre = path[i-1]
        post = path[i]
        ci = net.find_connection_index(pre, post)

        stride = int(net.connections[ci].s)
        cumulative_stride *= stride

        kernel_width = int(net.connections[ci].w)

        kernel = np.ones(kernel_width) / kernel_width
        rf = np.convolve(rf, kernel, mode='same')

        indices = indices[::stride]
        rf = rf[::stride]

        if plot: plt.plot(indices, rf)

    peak_rf_index = np.argmax(rf)
    within_one_sd = np.where(rf >= np.exp(-.5)*rf[peak_rf_index])[0]

    numerical_sd = input_pixel_width * (indices[within_one_sd[-1]] - indices[within_one_sd[0]]) / 2
    target_sd = system.find_population(path[-1]).w
    assert np.abs(numerical_sd - target_sd) < .2*target_sd, '{} vs {}'.format(numerical_sd, target_sd)

    if plot:
        print('numerical RF width: {} target RF width: {}'.format(numerical_sd, target_sd))
        plt.show()


def test_extrinsic_inputs():
    system = small_system()
    net = load_network()

    name = 'V4_4'
    expected_inputs = system.find_population(name).e

    total_inputs = 0
    for inbound in net.find_inbounds(name):
        total_inputs += inbound.pre.m * inbound.w**2 * inbound.c * inbound.sigma

    assert total_inputs == pytest.approx(expected_inputs, expected_inputs/10)


if __name__ == '__main__':
    optimize_and_save_network()
    test_stride()
    test_rf_size()
    test_extrinsic_inputs()