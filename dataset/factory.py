from domain.base import Factory
from dataset.mnist_datamodule import MNISTDataModule


class DataModuleFactory(Factory):

    def __init__(self):
        super(DataModuleFactory, self).__init__()

    @classmethod
    def create(cls, data_name):
        data_module = None

        if data_name == "mnist":
            data_module = MNISTDataModule

        return data_module
