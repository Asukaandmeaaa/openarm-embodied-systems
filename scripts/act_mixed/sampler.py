"""Episode/group balanced, deterministic source-weighted sampler."""
import numpy as np
import torch
from lerobot.datasets.sampler import EpisodeAwareSampler


class MixedEpisodeSampler(EpisodeAwareSampler):
    episode_masses = None

    def __init__(self,*args,**kwargs):
        masses=kwargs.pop('episode_masses',None)
        if masses is None:masses=type(self).episode_masses
        if masses is None:raise ValueError('No mixed sampling manifest configured')
        super().__init__(*args,**kwargs)
        from_indices=np.asarray(args[0]);to_indices=np.asarray(args[1])
        if len(masses)!=len(from_indices):raise ValueError('Sampler manifest episode count differs')
        selected=kwargs.get('episode_indices_to_use')
        selected=list(range(len(from_indices))) if selected is None else sorted(selected)
        weights=[]
        for ep in selected:
            length=int(to_indices[ep]-from_indices[ep])-kwargs.get('drop_n_first_frames',0)-kwargs.get('drop_n_last_frames',0)
            if length<=0:raise ValueError('A weighted episode has no valid frames')
            mass=float(masses[ep])
            if not np.isfinite(mass) or mass<=0:raise ValueError('Episode mass must be positive')
            weights.extend([mass/length]*length)
        self._sample_indices=self.indices
        self._weights=torch.tensor(weights,dtype=torch.float64)
        if len(weights)!=len(self._sample_indices):raise ValueError('Sampler length mismatch')

    def _iter_epoch(self,epoch,start):
        order=torch.multinomial(self._weights,self._num_frames,replacement=True,generator=self._epoch_generator(epoch))
        for k in range(start,self._num_frames):yield self._sample_indices[int(order[k])]
