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

TODO: do we need to enumerate all options or just try random ones?
TODO: This code assumes the network has a single input.
"""

import numpy as np
import networkx as nx
import calc.system


class Candidate:

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

        self.strides = [None for projection in system.projections]
        self.cumulatives = [None for population in system.populations]

        input_index = system.find_population_index(system.input_name)
        self.cumulatives[input_index] = 1

    def fill(self):
        """
        Fills in None strides in the network with candidate values.
        """

        while max([x is None for x in self.strides]):
            path = self._longest_unset_path()
            # print(path)

            start_cumulative = self.cumulatives[self.system.find_population_index(path[0])]
            end_cumulative = self.cumulatives[self.system.find_population_index(path[-1])]

            steps = len(path) - 1
            max_stride = Candidate._get_max_stride(self.max_cumulative_stride, steps)

            if end_cumulative is not None:
                if start_cumulative is not None:
                    max_stride = Candidate._get_max_stride(end_cumulative/start_cumulative, steps)
                else:
                    max_stride = Candidate._get_max_stride(end_cumulative, steps)

            # print('start c: {} end c: {} max stride: {} len: {}'.format(start_cumulative, end_cumulative, max_stride, len(path)-1))
            self.init_path(path, exact_cumulative=end_cumulative, max_stride=max_stride)

    @staticmethod
    def _get_max_stride(cumulative_stride, steps):
        return int(np.ceil(2 * cumulative_stride ** (1 / steps)))

    def _longest_unset_path(self):
        """
        :return: Longest path through the network that includes only connections for which
            the stride has not yet been determined for this Candidate
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

            for i in range(len(path) - 1):
                projection_ind = self.system.find_projection_index(path[i], path[i+1])
                if strides[projection_ind] is None:
                    strides[projection_ind] = np.random.randint(min_stride, max_stride+1)
                    pre_ind = self.system.find_population_index(path[i])
                    post_ind = self.system.find_population_index(path[i+1])
                    cumulatives[post_ind] = cumulatives[pre_ind] * strides[projection_ind]

            end_cumulative = cumulatives[self.system.find_population_index(path[-1])]

            if exact_cumulative is None or exact_cumulative == end_cumulative:
                if end_cumulative <= self.max_cumulative_stride:
                    self.strides = strides
                    self.cumulatives = cumulatives
                    done = True
                    break

        if not done:
            print('initialization failed')


if __name__ == '__main__':
    # system = calc.system.get_example_small()
    system = calc.system.get_example_medium()
    # path = longest_path(system, 'V4_5')
    # print(path)

    candidate = Candidate(system, 32)
    candidate.fill()
    print(candidate.strides)
    print(candidate.cumulatives)

    for i in range(len(system.populations)):
        print('{}: {}'.format(system.populations[i].name, candidate.cumulatives[i]))


    # candidate.init_path(path)
    #
    # for i in range(len(system.projections)):
    #     projection = system.projections[i]
    #     print('{}->{}: {}'.format(projection.origin.name, projection.termination.name, candidate.strides[i]))
    #
    # print(candidate.longest_unset_path())
