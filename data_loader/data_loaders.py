from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.datasets import PascalVOCDataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):
        trsfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm
        )
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )


class PascalVOCDataLoader(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        batch_size,
        keep_difficult=True,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):

        self.data_dir = data_dir
        self.dataset = PascalVOCDataset(
            data_folder=data_dir,
            split="train" if training else "test",
            keep_difficult=keep_difficult,
        )

        super().__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            num_workers,
            collate_fn=self.dataset.collate_fn,
        )
