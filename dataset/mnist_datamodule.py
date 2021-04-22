from typing import Optional, Union, List

from dataset.base import DataModuleBase
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(DataModuleBase):

    def __init__(self, *args, **kwargs):
        # DataModuleBase.convert_arguments(kwargs=kwargs)
        super(MNISTDataModule, self).__init__(*args, **kwargs)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dims = (1, 28, 28)

    def prepare_data(self, *args, **kwargs):
        # Download
        MNIST(root=self.data_dir, train=True, download=True)
        MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        stage = stage

        # Set stage
        if stage == "fit" or stage == "whole":
            mnist_full = MNIST(root=self.data_dir, train=True, transform=self.transform)
            self.train_dataset, self.val_dataset = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage == "whole":
            self.test_dataset = MNIST(root=self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.val_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.test_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size
        )
