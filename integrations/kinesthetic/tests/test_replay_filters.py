import numpy as np

from openarm_kinesthetic.replay.official_filters import BiquadLowpass, HermiteUpsampler


def test_hermite_hits_chunk_knots_and_biquad_preserves_constant() -> None:
    y=np.asarray([[0,1],[1,2],[2,3]],dtype=np.float32);up=HermiteUpsampler(2,1.0);out=up.upsample(y,np.asarray([0,.5,1.0]))
    assert np.allclose(out,y)
    filt=BiquadLowpass(1000,15);values=np.asarray([filt.step(np.asarray([2.0])) for _ in range(20)])
    assert np.allclose(values,2.0,atol=1e-5)
