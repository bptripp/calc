import numpy as np
import pytest
from calc.data import Data, areas_FV91


@pytest.fixture
def data():
    return Data()


def test_FLNe(data):
    n = len(areas_FV91)

    for target in areas_FV91:
        FLNe = np.zeros(n)
        for i in range(n):
            FLNe[i] = data.get_FLNe(areas_FV91[i], target)

        # all FLNe for target should sum to 1 unless there are no sources
        assert np.sum(FLNe) == pytest.approx(0, 1e-6) or np.sum(FLNe) == pytest.approx(1, 1e-6)


def test_population_size_reasonable(data):
    # These values are approximately products of area and #/mm^2 from Schmidt et al. (2018)
    # supplementary tables S8 and S9.
    assert data.get_num_neurons('V4', '2/3') == pytest.approx(35000000, 5000000)
    assert data.get_num_neurons('MT', '4') == pytest.approx(1500000, 500000)
    assert data.get_num_neurons('V1', '2/3') == pytest.approx(70000000, 5000000)

    # sanity checks for L4 sub-layers
    l4_total = 105000000
    for layer in ['4A', '4B', '4Calpha', '4Cbeta']:
        n = data.get_num_neurons('V1', layer)
        assert n < l4_total / 2
        assert n > l4_total / 15


def test_SLN(data):
    # sanity check for a few areas
    areas = ['V1', 'V3', 'V4', 'PITd']
    for pre in areas:
        for post in areas:
            SLN = data.get_SLN(pre, post)
            if np.isfinite(SLN):
                assert 0 <= SLN <= 100

def test_layers(data):
    actual_layers = data.get_layers('MT')
    for expected_layer in ['1', '2/3', '4', '5', '6']:
        assert expected_layer in actual_layers


def test_receptive_field_size(data):
    total_undefined = 0
    smallest = .5 # degrees visual angle
    biggest = 40
    for area in areas_FV91:
        size = data.get_receptive_field_size(area)
        if size is None:
            total_undefined += 1
        else:
            assert smallest <= size <= biggest

    assert total_undefined <= len(areas_FV91) - 10


def test_inputs_per_neuron(data):
    areas = ['V2', 'V4', 'MT']

    for area in areas:
        assert 100 < data.get_inputs_per_neuron(area, '4', '2/3') < 1000
        assert 100 < data.get_inputs_per_neuron(area, '2/3', '5') < 1000
        assert 50 < data.get_inputs_per_neuron(area, '5', '6') < 500


def test_extrinsic_inputs(data):
    areas = ['V2', 'V4', 'MT']

    for area in areas:
        assert 100 < data.get_extrinsic_inputs(area, '4') < 1000


def test_source_areas(data):
    sources = data.get_source_areas('V4', feedforward_only=True)
    assert 'V1' in sources
    assert 'V2' in sources
    assert 'MT' not in sources
    assert 'DP' not in sources
    assert 'PITd' not in sources
