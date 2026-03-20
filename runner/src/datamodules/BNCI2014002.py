"""DataModule for MOABB BNCI2014-002 (Motor Imagery EEG)."""

from sklearn.preprocessing import LabelEncoder
from moabb.datasets import BNCI2014_002
from moabb.paradigms import MotorImagery

from datamodules.base_datamodule import BaseEEGDataModule


class BNCI2014002DataModule(BaseEEGDataModule):
    def load_raw_data(self):
        paradigm = MotorImagery()
        data = paradigm.get_data(BNCI2014_002())

        # Groups are runs
        groups = data[2]["run"].values
        groups = LabelEncoder().fit_transform(groups)

        # No initial mask for this dataset
        ts = data[0]  # (n_trials, n_channels, n_timepoints)
        y = LabelEncoder().fit_transform(data[1])

        # Remove first run (group 0) as in DiffeoCFM
        mask = groups != 0
        ts, y, groups = ts[mask], y[mask], groups[mask]
        groups = groups - 1

        return ts, y, groups
