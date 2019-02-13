# make sure strides are consistent

import pytest
from calc.examples.example_systems import make_big_system, StridePattern


@pytest.fixture
def system():
    cortical_areas = ['V1', 'V2', 'V4', 'VOT', 'PITd', 'PITv', 'CITd', 'CITv']
    return make_big_system(cortical_areas=cortical_areas)


def test_strides_consistent(system):
    candidate = StridePattern(system, 32)
    candidate.fill()

    for i in range(len(system.projections)):
        stride = candidate.strides[i]

        pre = system.projections[i].origin.name
        pre_index = system.find_population_index(pre)
        pre_cumulative = candidate.cumulatives[pre_index]

        post = system.projections[i].termination.name
        post_index = system.find_population_index(post)
        post_cumulative = candidate.cumulatives[post_index]

        # print('{}->{} {}-{}-{}'.format(
        #     pre, post, pre_cumulative, stride, post_cumulative))

        assert pre_cumulative * stride == post_cumulative, 'pre: {} post: {}'.format(pre, post)


def test_strides_not_too_big(system):
    max_cumulative = 32
    candidate = StridePattern(system, max_cumulative)
    candidate.fill()

    for i in range(len(system.populations)):
        cumulative = candidate.cumulatives[i]

        assert cumulative <= max_cumulative, \
            'population: {} cumulative: {}'.format(system.populations[i], cumulative)

