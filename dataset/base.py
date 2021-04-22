from abc import abstractmethod
from argparse import Namespace
from idlelib.config import _warn
from typing import Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.utilities import AttributeDict, rank_zero_only

from properties import APPLICATION_PROPERTIES


rank_zero_warn = rank_zero_only(_warn)


class DataModuleBase(pl.LightningDataModule):

    def __init__(self, *args, **kwargs):
        DataModuleBase.convert_arguments(kwargs=kwargs)
        super(DataModuleBase, self).__init__()
        self.hparam_dict = kwargs

        # Hyperparameters
        self.data_dir = self.hparam_dict["data_dir"]
        self.batch_size = self.hparam_dict["batch_size"]
        self.num_workers = self.hparam_dict["num_workers"]
        self.pin_memory = self.hparam_dict["pin_memory"]

    @abstractmethod
    def prepare_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        pass

    @classmethod
    def convert_arguments(cls, kwargs):
        arg_dict = kwargs

        if arg_dict["data_dir"] == "auto":
            arg_dict["data_dir"] = APPLICATION_PROPERTIES.DATA_DIRECTORY_PATH
