"""DataModule for MOABB BNCI2015-001 (Motor Imagery EEG)."""

from sklearn.preprocessing import LabelEncoder
from moabb.datasets import BNCI2015_001
from moabb.paradigms import MotorImagery

from src.datamodules.base_datamodule import BaseEEGDataModule


class BNCI2015001DataModule(BaseEEGDataModule):
    def load_raw_data(self):
        paradigm = MotorImagery()
        data = paradigm.get_data(BNCI2015_001())

        # Groups are sessions; remove 3rd session ("2C")
        groups = data[2]["session"].values
        mask = groups != "2C"

        ts = data[0][mask]  # (n_trials, n_channels, n_timepoints)
        y = LabelEncoder().fit_transform(data[1][mask])
        groups = LabelEncoder().fit_transform(groups[mask])

        return ts, y, groups
