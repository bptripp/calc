"""
This file contains code related to setting stride hyperparameters. The other
hyperparameters are optimized using gradient descent, and rounded if they are
supposed to be integers. However, this doesn't work well for strides. Because
they are integers that have small optimal values, optimizing as floats and rounding
can cause problems such as large quantization error, inconsistency between
converging paths, and strides of 0.

To avoid these problems, we choose random candidate stride patterns, and then
optimize other parameters using gradient descent. We partially optimize the
other hyperparameters of a number of candidates and choose the one with the best
result. The algorithm is as follows:

1: Start by setting all strides to None
2: While at least one stride is None:
2.1: Find longest path within network that consists only of edges with None strides
2.2: Set strides along longest path
2.3: If cumulative stride along path is greater than image width or doesn't equal
    gain from start to end (if cumulative strides at each end are known) return to 2.1

Note this code assumes the network has a single input.
"""

import numpy as np
import networkx as nx
import calc.conversion
import calc.system, calc.network
from calc.data import areas_FV91, E07


def get_stride_pattern(system, max_cumulative_stride=512, best_of=10):
    """
    :param system: system.System (biological network parameters)
    :param max_cumulative_stride: largest allowable cumulative stride, typically smallest image dimension
    :param best_of: number of random patterns to try
    :return: best stride pattern found; list of cost of different attempts relative to hints from
        physiology; first several examples
    """
    best_distance = 1e10
    best_pattern = None
    distances = []
    first_few = []

    for i in range(best_of):
        print('Making stride pattern {} of {}'.format(i, best_of))
        candidate = StridePattern(system, max_cumulative_stride)
        # candidate.set_hints()
        candidate.fill()
        distance = candidate.distance_from_hints()
        distances.append(distance)
        if distance < best_distance:
            best_distance = distance
            best_pattern = candidate
        if i < 6:
            first_few.append(candidate)

    return best_pattern, distances, first_few


class StridePattern:

    def __init__(self, system, max_cumulative_stride):
        """
        Initializes a stride-pattern candidate with null strides.

        :param system: the System for which candidate strides are to be proposed
        :param max_cumulative_stride: maximum cumulative stride through the system along
            any feedforward path; this would nornally be the resolution of the
            input image, to prevent later feature maps from having less than
            one-pixel resolution
        """

        self.system = system
        self.max_cumulative_stride = max_cumulative_stride
        self._reset()

    def _reset(self):
        self.strides = [None] * len(system.projections)
        self.cumulatives = [None] * len(system.populations)
        self.cumulative_hints = [None] * len(system.populations)
        self.min_cumulatives = [1] * len(system.populations)
        self.max_cumulatives = [self.max_cumulative_stride] * len(system.populations)
        input_index = system.find_population_index(system.input_name)
        self.cumulatives[input_index] = 1
        self.set_hints()


    def set_hints(self, image_layer=0, image_channels=3, V1_channels=130, other_channels={'LGNparvo': 4, 'LGNmagno': 2, 'LGNkonio': 1}):
        """
        Sets hints from physiology about cumulative strides in each layer. These are based on
        the idea that the number of channels in a layer should scale with the density of spines on
        basal dendrites (see rationale in paper).

        :param image_layer: index of network layer that corresponds to input image
        :param image_channels: number of channels in image (typically 3)
        :param V1_channels: suggested number of channels in V1 layers (see rationale in paper)
        :param other_channels: manual suggestions for subcortical channels
        """

        image_pixels = np.sqrt(system.populations[image_layer].n / image_channels)

        e07 = E07()
        V1_spine_count = e07.get_spine_count('V1')

        for i in range(len(self.system.populations)):
            pop = self.system.populations[i]
            area = pop.name.split('_')[0]
            if i == image_layer:
                channels = image_channels
            if pop.name in other_channels.keys():
                channels = other_channels[pop.name]
            elif area in areas_FV91 and '2/3' in pop.name:
                spine_count = e07.get_spine_count(area)
                channels = np.round(V1_channels * spine_count / V1_spine_count)
            else:
                channels = None

            if channels:
                pixels = np.sqrt(pop.n / channels)
                self.cumulative_hints[i] = image_pixels / pixels

    def distance_from_hints(self):
        """
        :return: A cost function based on the differences between cumulative strides and corresponding
            physiological hints (root of mean of squared log-ratios)
        """

        total = 0
        count = 0
        for i in range(len(self.cumulative_hints)):
            if self.cumulative_hints[i]:
                error = np.log(self.cumulatives[i] / self.cumulative_hints[i])**2
                total += error
                count += 1
        return np.sqrt(total / count)

    def _update_cumulative_stride_bounds(self):
        graph = self.system.make_graph()

        for i in range(len(self.system.populations)):
            pop = self.system.populations[i]

            # cumulative stride can't be less than that of ancestors
            ancestors = nx.ancestors(graph, pop.name)
            for ancestor in ancestors:
                ancestor_index = self.system.find_population_index(ancestor)
                if self.cumulatives[ancestor_index] is not None:
                    self.min_cumulatives[i] = max(self.min_cumulatives[i], self.cumulatives[ancestor_index])

            # cumulative stride can't be greater than that of descendants
            descendants = nx.descendants(graph, pop.name)
            for descendant in descendants:
                descendant_index = self.system.find_population_index(descendant)
                if self.cumulatives[descendant_index] is not None:
                    self.max_cumulatives[i] = min(self.max_cumulatives[i], self.cumulatives[descendant_index])

    def fill(self):
        """
        Fills in None strides in the network with candidate values.
        """

        while max([x is None for x in self.strides]):
            path = self._longest_unset_path()
            # print('Setting strides for path: {}'.format(path))

            start_index = self.system.find_population_index(path[0])
            end_index = self.system.find_population_index(path[-1])

            start_cumulative = self.cumulatives[start_index]
            end_cumulative = self.cumulatives[end_index]

            low_cumulative = start_cumulative if start_cumulative else self.min_cumulatives[start_index]
            high_cumulative = end_cumulative if end_cumulative else self.max_cumulatives[end_index]

            steps = len(path) - 1
            max_stride = StridePattern._get_max_stride(high_cumulative / low_cumulative, steps)

            # print('start c: {} end c: {} max c: {} max stride: {} len: {}'.format(
            #     start_cumulative, end_cumulative, self.max_cumulative_stride, max_stride, len(path)-1))
            success = self.init_path(path, exact_cumulative=end_cumulative, max_stride=max_stride)

            if not success:
                print('Resetting all strides')
                self._reset()

            self._update_cumulative_stride_bounds()

    @staticmethod
    def _get_max_stride(cumulative_stride, steps):
        # ceil produces slightly better results but is substantially slower
        return int(2 * np.floor(cumulative_stride ** (1 / steps)))

    def _longest_unset_path(self):
        """
        :return: Longest path through the network that includes only connections for which
            the stride has not yet been determined for this StridePattern
        """
        graph = self.system.make_graph()

        for i in range(len(self.system.projections)):
            if self.strides[i] is not None:
                origin = self.system.projections[i].origin.name
                termination = self.system.projections[i].termination.name
                graph.remove_edge(origin, termination)

        return nx.algorithms.dag.dag_longest_path(graph)

    def init_path(self, path, exact_cumulative=None, min_stride=1, max_stride=3, max_attempts=10000):
        """
        Sets strides along the path to integer values between min_stride and
        max_stride. Strides are sampled at random, and rejected if the
        cumulative stride along the path (product of all strides) is greater than
        max_cumulative, and/or not equal to exact_cumulative (if this is not None).
        This method does not change strides that have been set previously.

        :param path: a path, in the form of a list of node names, along which to choose random strides
        :param exact_cumulative (default None): if not None, defines the exact cumulative stride
            required at the end of the path (for consistency with other strides in the network)
        :param min_stride (default 1): minimum random stride value
        :param max_stride (default 3): maximum random stride value
        """

        done = False

        for attempt in range(max_attempts):
            # make copies in case we have to revert
            strides = self.strides[:]
            cumulatives = self.cumulatives[:]

            failed = False

            for i in range(len(path) - 1):
                projection_ind = self.system.find_projection_index(path[i], path[i+1])
                pre_ind = self.system.find_population_index(path[i])
                post_ind = self.system.find_population_index(path[i+1])

                if self.cumulatives[post_ind] and self.cumulatives[pre_ind]:
                    strides[projection_ind] = self.cumulatives[post_ind] / self.cumulatives[pre_ind]
                    if abs(strides[projection_ind] - round(strides[projection_ind])) > 1e-3:
                        # this can happen e.g. if pre is 2 and post is 3 (have to start over then)
                        failed = True
                        break
                else:
                    max_for_this_stride = min(max_stride, int(self.max_cumulatives[post_ind]/cumulatives[pre_ind]))
                    min_for_this_stride = max(min_stride, int(self.min_cumulatives[post_ind]/cumulatives[pre_ind]))

                    if max_for_this_stride < min_for_this_stride:
                        failed = True
                        break

                    strides[projection_ind] = self._sample_stride(min_for_this_stride, max_for_this_stride)
                    cumulatives[post_ind] = cumulatives[pre_ind] * strides[projection_ind]

                    if cumulatives[post_ind] > self.max_cumulatives[post_ind] \
                            or cumulatives[post_ind] < self.min_cumulatives[post_ind]:
                        # this can happen due to rounding in min_for_this_stride
                        failed = True
                        break

            if not failed:
                end_cumulative = cumulatives[self.system.find_population_index(path[-1])]
                if exact_cumulative is None or exact_cumulative == end_cumulative:
                    self.strides = strides
                    self.cumulatives = cumulatives
                    done = True
                    break

        if not done:
            print(path)
            print('initialization failed; exact cumulative {}, min {}, max {}'.format(exact_cumulative, min_stride, max_stride))

        return done

    def _sample_stride(self, min_stride, max_stride, stride_hint=1.25):
        possible_strides = range(min_stride, max_stride + 1)

        relative_probabilities = [1/(.1+np.abs(stride-stride_hint))**2 for stride in possible_strides]
        probabilities = relative_probabilities / np.sum(relative_probabilities)
        result = np.random.choice(possible_strides, p=probabilities)

        return result


def initialize_network(system, candidate, image_layer=0, image_channels=3.):
    """
    :param system: system.System (biological network parameters)
    :param candidate: a plausible StridePattern
    :param image_layer: index of system layer that corresponds to input image
    :param image_channels: number of channels in input image
    :return: A neural network architecture with the same nodes and connections as the given
        neurophysiological system architecture, the given stride pattern, with other
        hyperparameters initialized randomly.
    """
    net = calc.network.Network()

    approx_image_resolution = np.sqrt(system.populations[image_layer].n/image_channels)
    max_cumulative_stride = np.max(candidate.cumulatives)
    image_resolution = round(approx_image_resolution / max_cumulative_stride) * max_cumulative_stride

    for i in range(len(system.populations)):
        pop = system.populations[i]

        if i == image_layer:
            channels = image_channels
            pixels = image_resolution
        else:
            pixels = image_resolution / candidate.cumulatives[i]
            channels = max(1, round(pop.n / pixels**2))

        net.add(pop.name, channels, pixels)

    for i in range(len(system.projections)):
        projection = system.projections[i]
        pre = net.find_layer(projection.origin.name)
        post = net.find_layer(projection.termination.name)

        stride = candidate.strides[i]

        c = .1 + .2*np.random.rand()
        sigma = .1 + .1*np.random.rand()

        # in optimization this value is discarded and the value is calculated from
        # RF widths, which are optimized
        w = 7

        net.connect(pre, post, c, stride, w, sigma)

    return net


if __name__ == '__main__':
    from calc.examples.example_systems import make_big_system, make_small_system
    import pickle

    cortical_areas = ['V1', 'V2', 'V4', 'VOT', 'PITd', 'PITv', 'CITd', 'CITv', 'AITd', 'AITv']
    system = make_big_system(cortical_areas)

    system.prune_FLNe(0.15)
    system.normalize_FLNe()
    # system.print_description()
    # print(len(system.projections))

    # system = calc.system.get_example_small()
    # system = calc.system.get_example_medium()
    # path = longest_path(system, 'V4_5')
    # print(path)

    candidate, distances, first_few = get_stride_pattern(system, best_of=1000)
    with open('stride-pattern.pkl', 'wb') as file:
        pickle.dump({'system': system, 'strides': candidate, 'distances': distances, 'first_few': first_few}, file)

    # with open('stride-pattern.pkl', 'rb') as file:
    #     result = pickle.load(file)
    # import matplotlib.pyplot as plt
    # plt.hist(result['distances'])
    # plt.show()

    # print(candidate.strides)
    print(candidate.cumulatives)

    for i in range(len(system.populations)):
        print('{}: {} vs {}'.format(system.populations[i].name, candidate.cumulatives[i], candidate.cumulative_hints[i]))
    #
    # print('distance from hints: {}'.format(candidate.distance_from_hints()))

    net = initialize_network(system, candidate, image_layer=0, image_channels=3.)
    net.print()

    # net, training_curve = calc.conversion.test_stride_pattern(system)
    # import matplotlib.pyplot as plt
    # plt.semilogy(training_curve)
    # plt.show()

    # calc.conversion.test_stride_patterns(system)

    # candidate.init_path(path)
    #
    # for i in range(len(system.projections)):
    #     projection = system.projections[i]
    #     print('{}->{}: {}'.format(projection.origin.name, projection.termination.name, candidate.strides[i]))
    #
    # print(candidate.longest_unset_path())

