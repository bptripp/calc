"""
The stride parameters of connections benefits from special treatment, because they
are integers that have small optimal values. Rounding them after optimizing as floats
can cause large changes, inconsistency between converging paths, and strides of 0.

To avoid these problems, we first choose a random candidate stride pattern, and then
optimize other parameters as float (using gradients).

First:
- find longest path from input to output

To produce a candidate stride pattern:
1. set strides along longest path between 1 and 3 at random
2. if cumulative stride is greater than image width (<1 pixel at output) return to 1
    (note this check vastly reduces the candidates for networks with realistic depth)
3. for each remaining connection in random order:
    if cumulative stride of origin and termination are set, stride is their ratio; if
        ratio <1 restart 3
    otherwise set stride between 1 and 3 at random
"""

import numpy as np
import networkx as nx
import calc.system


def longest_path(system, output_name):
    """
    :param system:
    :param output_name:
    :return:
    """
    G = system.make_graph()

    """
    The rest of this function is copied from networkx.algorithms.dag.dag_longest_path, 
    with a small change to return the longest path that ends at a specified node
    rather than the longest path overall. The networkx code has the following license: 
    
    Copyright (C) 2004-2012, NetworkX Developers
    Aric Hagberg <hagberg@lanl.gov>
    Dan Schult <dschult@colgate.edu>
    Pieter Swart <swart@lanl.gov>
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:
    
      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
    
      * Redistributions in binary form must reproduce the above
        copyright notice, this list of conditions and the following
        disclaimer in the documentation and/or other materials provided
        with the distribution.
    
      * Neither the name of the NetworkX Developers nor the names of its
        contributors may be used to endorse or promote products derived
        from this software without specific prior written permission.
    
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     
    """
    dist = {}  # stores [node, distance] pair
    for node in nx.topological_sort(G):
        # pairs of dist,node for all incoming edges
        pairs = [(dist[v][0] + 1, v) for v in G.pred[node]]
        if pairs:
            dist[node] = max(pairs)
        else:
            dist[node] = (0, node)
    node, (length, _) = output_name, dist[output_name] # original code: node, (length, _) = max(dist.items(), key=lambda x: x[1])

    path = []
    while length > 0:
        path.append(node)
        length, node = dist[node]
    return list(reversed(path))


def none_max(a):
    """
    :param a: list containing numbers and None
    :return: max of the numbers
    """
    max = -np.inf
    for item in a:
        if item is not None and item > max:
            max = item
    return max


class Candidate:
    # TODO: This code assumes the network has a single input.

    def __init__(self, system):
        """
        Initializes a stride-pattern candidate with null strides.

        :param system: the System for which strides are to be proposed
        """

        self.system = system
        self.strides = [None for projection in system.projections]
        self.cumulatives = [None for population in system.populations]

        input_index = system.find_population_index(system.input_name)
        self.cumulatives[input_index] = 1
            

    def init_path(self, path, max_cumulative, min_stride=1, max_stride=3, max_attempts=10000):
        """
        Sets strides along the path to integer values between min_stride and
        max_stride. A list of strides is sampled at random, and rejected if the
        cumulative stride along the path (product of all strides) is greater than
        max_cumulative.

        This method should be called once for each output of the network. This call
        does not change strides that are set before the call.

        :param path: a path (list of node names) along which to choose random strides
        :param max_cumulative: maximum cumulative stride through the path
        """

        for attempt in range(max_attempts):
            # print('attempt: {}'.format(attempt))

            # make copies in case we have to revert
            strides = self.strides[:]
            cumulatives = self.cumulatives[:]

            for i in range(len(path) - 1):
                projection_ind = self.system.find_projection_index(path[i], path[i+1])
                if strides[projection_ind] is None:
                    strides[projection_ind] = np.random.randint(min_stride, max_stride)

                    pre_ind = self.system.find_population_index(path[i])
                    post_ind = self.system.find_population_index(path[i+1])
                    cumulatives[post_ind] = cumulatives[pre_ind] * strides[projection_ind]

            if none_max(cumulatives) <= max_cumulative:
                self.strides = strides
                self.cumulatives = cumulatives
                break

        if max(cumulatives) > max_cumulative:
            print('initialization failed')

    def fill(self, max_attempts=10000):
        """
        To be called after initializing longest paths to all outputs. This
        method fills in remaining strides.

        TODO: not sure how to efficiently update cumulatives if we fill
        middle of multi-step path -- maybe wait until end and fill any missing
        by stepping forward -- can't really since
        """
        for attempt in range(max_attempts):
            strides = self.strides[:]
            cumulatives = self.cumulatives[:]

            remaining_ind = []
            for i in range(len(strides)):
                if strides[i] is None:
                    remaining_ind.append(i)

            while len(remaining_ind) > 0:
                pass


# def make_net_from_system(system, image_layer=0, image_channels=3.):
#     """
#     :return: A neural network architecture with the same nodes and connections as the given
#         neurophysiological system architecture, and otherwise random hyperparameters
#         (to be optimized separately)
#     """
#     net = Network()
#
#     for i in range(len(system.populations)):
#         pop = system.populations[i]
#         units = pop.n
#
#         if i == image_layer:
#             channels = image_channels
#             pixels = round(np.sqrt(units/image_channels))
#         else:
#             ratio_channels_over_pixels = np.exp(-1.5 + 3*np.random.rand())
#             pixels = round(np.cbrt(units / ratio_channels_over_pixels))
#             channels = round(ratio_channels_over_pixels * pixels)
#         net.add(pop.name, channels, pixels)
#
#     # try to set strides to reasonable values
#     for i in range(len(system.populations)):
#         pres = system.find_pre(system.populations[i].name)
#         units = net.layers[i].m * net.layers[i].width**2
#
#         if len(pres) > 0:
#             pre_ind = system.find_population_index(pres[0].name)
#             net.layers[i].width = net.layers[pre_ind].width / 1.5
#             net.layers[i].m = units / net.layers[i].width**2
#
#     for projection in system.projections:
#         pre = net.find_layer(projection.origin.name)
#         post = net.find_layer(projection.termination.name)
#
#         # c = projection.f
#         c = .1 + .2*np.random.rand()
#         s = 1. + 9.*np.random.rand()
#         rf_ratio = projection.termination.w / projection.origin.w
#         w = (rf_ratio - 1.) / (0.5 + np.random.rand())
#         w = np.maximum(.1, w)  # make sure kernel has +ve width
#         sigma = .1 + .1*np.random.rand()
#         net.connect(pre, post, c, s, w, sigma)
#
#     return net

if __name__ == '__main__':
    system = calc.system.get_example_small()
    path = longest_path(system, 'V4_5')
    print(path)

    candidate = Candidate(system)
    candidate.init_path(path, 32)

    for i in range(len(system.projections)):
        projection = system.projections[i]
        print('{}->{}: {}'.format(projection.origin.name, projection.termination.name, candidate.strides[i]))

    # print(candidate.strides)