"""Local LeRobot reader optimization; byte/tensor-identical target windows.

HF Column indexing applies the full custom transform in the installed version,
decoding cameras even for an action-only window. Narrow columns BEFORE rows.
This module does not modify site-packages, indices, padding or normalization.
"""
import torch
from lerobot.datasets.dataset_reader import DatasetReader


class SelectedColumnReader(DatasetReader):
    def _query_hf_dataset(self,query_indices):
        result={}
        for key,indices in query_indices.items():
            if key in self._meta.video_keys:continue
            relative=(indices if self._absolute_to_relative_idx is None else
                      [self._absolute_to_relative_idx[i] for i in indices])
            result[key]=torch.stack(self.hf_dataset.select_columns([key])[relative][key])
        return result


def enable_selected_columns(dataset):
    reader=dataset._ensure_reader()
    if not isinstance(reader,SelectedColumnReader):
        if type(reader) is not DatasetReader:
            raise ValueError('Unrecognized LeRobot reader; verify column optimization compatibility')
        # Same Python instance/state; a top-level subclass remains pickleable
        # by spawn workers, unlike replacing an instance method with a closure.
        reader.__class__=SelectedColumnReader
    return dataset
