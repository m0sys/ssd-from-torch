from math import sqrt

import torch.nn as nn
import torch.nn.functional as F
import torch

from base import BaseModel
from utils.util import find_jaccard_overlap, cxcy_to_xy, gcxgcy_to_cxcy
from model._model import *


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class SSD300(BaseModel):
    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes

        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)

        # Scaling factor for lower level featues (i.e. conv4_3_feats) - learned parameter.
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def create_prior_boxes(self):
        """
        Create the 8732 prior (default/anchor) boxes for the SSD300, as defined in the paper.
        https://arxiv.org/abs/1512.02325

        Returns:
            prior boxes in center-size coordinates, i.e. a tensor of dim (8732, 4)
        """
        fmap_dims = {
            "conv4_3": 38,
            "conv7": 19,
            "conv8_2": 10,
            "conv9_2": 5,
            "conv10_2": 3,
            "conv11_2": 1,
        }

        obj_scales = {
            "conv4_3": 0.1,
            "conv7": 0.2,
            "conv8_2": 0.375,
            "conv9_2": 0.55,
            "conv10_2": 0.725,
            "conv11_2": 0.9,
        }

        aspect_ratios = {
            "conv4_3": [1.0, 2.0, 0.5],
            "conv7": [1.0, 2.0, 3.0, 0.5, 0.333],
            "conv8_2": [1.0, 2.0, 3.0, 0.5, 0.333],
            "conv9_2": [1.0, 2.0, 3.0, 0.5, 0.333],
            "conv10_2": [1.0, 2.0, 0.5],
            "conv11_2": [1.0, 2.0, 0.5],
        }

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append(
                            [
                                cx,
                                cy,
                                obj_scales[fmap] * sqrt(ratio),
                                obj_scales[fmap] / sqrt(ratio),
                            ]
                        )

                        # For an aspect ratio of 1, use an additional prior whose scale
                        # is the geometric mean of the scale of the current feature map
                        # and the scale of the next feature map.

                        if ratio == 1.0:
                            try:
                                additional_scale = self._geometric_mean(
                                    obj_scales[fmap], obj_scales[fmaps[k + 1]]
                                )

                            except IndexError:
                                additional_scale = 1.0

                            prior_boxes.append(
                                [cx, cy, additional_scale, additional_scale]
                            )

        prior_boxes = torch.FloatTensor(prior_boxes)  ## .to(device)  # (8732, 4)
        prior_boxes.clamp(0, 1)
        assert prior_boxes.shape == (8732, 4)
        return prior_boxes

    def _geometric_mean(self, x, y):
        return sqrt(x * y)

    def forward(self, x):
        """
        Args:
            a tensor of dimensions (N, 3, 300, 300) where N is the batch size.

        Returns:
            8732 locations and class scores for x.
        """
        bs = x.size(0)

        # Apply base network (i.e. Vgg).
        conv4_3_feats, conv7_feats = self.base(x)
        assert conv4_3_feats.shape == (bs, 512, 38, 38)
        assert conv7_feats.shape == (bs, 1024, 19, 19)

        # Rescale conv4_3_feats after L2 norm.
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()
        conv4_3_feats = conv4_3_feats / norm
        conv4_3_feats = conv4_3_feats * self.rescale_factors
        assert conv4_3_feats.shape == (bs, 512, 38, 38)

        # Apply aux network.
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(
            conv7_feats
        )
        assert conv8_2_feats.shape == (bs, 512, 10, 10)
        assert conv9_2_feats.shape == (bs, 256, 5, 5)
        assert conv10_2_feats.shape == (bs, 256, 3, 3)
        assert conv11_2_feats.shape == (bs, 256, 1, 1)

        # Apply predictor network.
        locs, cls_scores = self.pred_convs(
            conv4_3_feats,
            conv7_feats,
            conv8_2_feats,
            conv9_2_feats,
            conv10_2_feats,
            conv11_2_feats,
        )
        assert locs.shape == (bs, 8732, 4)
        assert cls_scores.shape == (bs, 8732, self.n_classes)

        return locs, cls_scores

    def detect_objects(
        self, predicted_locs, predicted_scores, min_score, max_overlap, top_k, device
    ):
        """
        Decipher the 8732 locations and class scores (output of this SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a min
        threshold.

        Args:
            predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes -
            tensor of dim (N, 8732, 4).

            predicted_scores: class scores for each of the encoded locations/boxes -
            tensor of dim (N, 8732, n_classes).

            min_score: min threshold for a box to be considered a match for a certain class.

            max_overlap: max overlap two boxes can have so that the one with the lower
            score is not suppressed via NMS.

            top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'.

        Returns:
            Detections (boxes, labels, and scores) - lists of length batch_size.
        """
        bs = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)
        assert predicted_scores.shape == (bs, 8732, self.n_classes)

        all_images_boxes = []
        all_images_labels = []
        all_images_scores = []

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        # TODO: Make this piece of code faster.
        for i in range(bs):
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy)
            )

            assert decoded_locs.shape == (8732, 4)

            image_boxes = []
            image_labels = []
            image_scores = []

            for c in range(1, self.n_classes):
                class_scores = predicted_scores[i][:, c]
                score_above_min_score = class_scores > min_score
                n_above_min_score = score_above_min_score.sum().item()

                if n_above_min_score == 0:
                    continue

                class_scores = class_scores[score_above_min_score]
                class_decoded_locs = decoded_locs[score_above_min_score]

                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
                class_decoded_locs = class_decoded_locs[sort_ind]

                suppress = self.nms(
                    class_decoded_locs, max_overlap, n_above_min_score, device
                )

                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(
                    torch.LongTensor((1 - suppress).sum().item() * [c]).to(device)
                )
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'.
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0.0, 0.0, 1.0, 1.0]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.0]).to(device))

            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects.
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores

    def nms(self, class_decoded_locs, max_overlap, n_above_min_score, device):
        """
        Non-Maximum Supreession of redundent predictions with an IOU greater than
        max_overlap.
        """

        overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)
        suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)

        for box in range(class_decoded_locs.size(0)):
            if suppress[box] == 1:
                continue

            condition = torch.tensor(overlap[box] > max_overlap, dtype=torch.uint8).to(
                device
            )
            suppress = torch.max(suppress, condition)

            # Don't suppess this box (overlap of 1 with itself).
            suppress[box] = 0

        return suppress
