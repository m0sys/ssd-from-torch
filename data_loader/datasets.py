import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image

from utils.trms_util import transform


class PascalVOCDataset(Dataset):
    """
    A Dataset class to be used in a DataLoader to create batches.
    """

    def __init__(self, data_folder, split="TEST", keep_difficult=False):
        """
        Args:
          data_folder: folder where data files are stored.
          split: split, one of 'TRAIN' or 'TEST'.
          keep_difficult: keep or discard objects that are condidered difficult to detect.
        """
        self.split = split.upper()

        assert self.split in {"TRAIN", "TEST"}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        with open(os.path.join(data_folder, self.split + "_images.json"), "r") as j:
            self.images = json.load(j)

        with open(os.path.join(data_folder, self.split + "_objects.json"), "r") as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        image = Image.open(self.images[i], mode="r")
        image = image.convert("RGB")

        objects = self.objects[i]
        boxes = torch.FloatTensor(objects["boxes"])
        labels = torch.LongTensor(objects["labels"])
        difficulties = torch.ByteTensor(objects["difficulties"])

        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]

        image, boxes, labels, difficulties = transform(
            image, boxes, labels, difficulties, split=self.split
        )

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Describes how to combine images with different number of objects by using lists.


        Since each image may have a different number of objects, we need a collate function
        (to bew passed to the DataLoader).

        Args:
          batch: an iterable of N sets from __getitem__()

        Returns:
          a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties.
        """
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties