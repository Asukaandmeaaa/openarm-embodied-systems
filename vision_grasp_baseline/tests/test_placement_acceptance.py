import pytest

from vision_grasp_baseline.place_test import inside_aperture


def test_reported_near_wall_release_is_accepted_as_inside():
    assert inside_aperture([.04402989484209372,.05696853422600874],
                           [.12036801299426629,.11646454174369185])


@pytest.mark.parametrize('extent,inner',[
    ([.061,.04],[.12,.12]),
    ([.04,.061],[.12,.12]),
    ([float('nan'),.04],[.12,.12]),
    ([.04,.04],[0.,.12]),
    ([-.01,.04],[.12,.12]),
])
def test_outside_or_invalid_observation_is_not_accepted(extent,inner):
    assert not inside_aperture(extent,inner)
