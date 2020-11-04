import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import cxcy_to_xy, xy_to_cxcy, cxcy_to_gcxgcy, find_jaccard_overlap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nll_loss(output, target):
    return F.nll_loss(output, target)


def multibox_loss(output, target):
    pass


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss - a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicated locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.0):
        super().__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Args:
            predicted_locs: predicted locations/boxes w.r.t. the 8732 prior boxes
                            i.e. a tensor of dims (N, 8732, 4).

            predicted_scores: class scores for each of the encoded locations/boxes
                              i.e. a tensor of dims (N, 8732, n_classes)

            boxes: true object bounding boxes in boundary coords,
                   i.e. a list of N tensors.

            labels: true object labels,
                    i.e. a list of N tensors.

        Returns:
            the Multibox loss which is a scalar.
        """
        bs = predicted_locs.size(0)
        npriors = self.priors_cxcy.size(0)
        ncls = predicted_scores.size(2)

        assert npriors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((bs, npriors, 4), dtype=torch.float).to(device)
        true_clss = torch.zeros((bs, npriors), dtype=torch.long).to(device)
        assert true_locs.shape == (bs, 8732, 4)
        assert true_clss.shape == (bs, 8732)

        for i in range(bs):

            n_objs = boxes[i].size(0)

            iou_bboxes_priors = find_jaccard_overlap(boxes[i], self.priors_xy)
            assert iou_bboxes_priors.shape == (n_objs, 8732)

            true_clss[i] = self.find_true_label_for_each_prior(
                labels[i], iou_bboxes_priors
            )
            true_locs[i] = self.find_true_bbox_for_each_prior(
                boxes[i], iou_bboxes_priors, n_objs
            )

        positive_priors = true_clss != 0

        loc_loss = self.calc_loc_loss(predicted_locs, true_locs, positive_priors)

        conf_loss = self.calc_confidence_loss(
            predicted_scores, true_clss, positive_priors, ncls, npriors, bs
        )

        return conf_loss + self.alpha * loc_loss

    def find_true_label_for_each_prior(self, label, iou_bboxes_priors):
        max_iou_for_each_prior, obj_at_max_iou = iou_bboxes_priors.max(dim=0)
        assert max_iou_for_each_prior.shape == (8732,)
        assert obj_at_max_iou.shape == (8732,)

        labels_for_each_prior = label[obj_at_max_iou]
        assert labels_for_each_prior.shape == (8732,)

        labels_for_each_prior[max_iou_for_each_prior < self.threshold] = 0
        assert labels_for_each_prior.shape == (8732,)

        return labels_for_each_prior

    def find_true_bbox_for_each_prior(self, box, iou_bboxes_priors, n_objs):
        max_iou_for_each_prior, obj_at_max_iou = iou_bboxes_priors.max(dim=0)
        assert max_iou_for_each_prior.shape == (8732,)
        assert obj_at_max_iou.shape == (8732,)

        self.make_obj_non_background(
            iou_bboxes_priors, max_iou_for_each_prior, obj_at_max_iou, n_objs
        )

        return self.encode_gt_bbox(box, obj_at_max_iou)

    def make_obj_non_background(
        self, iou_bboxes_priors, max_iou_for_each_prior, obj_at_max_iou, n_objs
    ):
        """
        Make all objects have an IOU of 1 with their corresponding prior with max-overlap.

        We don't want a situation where an object is not represented in our
        pos (non-bg) priors.

        This can happen in the two following situations:
        1. An object might not be the best object for all priors,
           and is therefore not in object_for_each_prior.
        2. All priors with the object may be assigned as background based
           on the threshold (0.5).

        To remdy this we assign each object to the max-overlaping prior and
        set these priors to have an artifical IOU of 1.
        """

        _, prior_at_max_iou = iou_bboxes_priors.max(dim=1)

        # Fix situation #1
        obj_at_max_iou[prior_at_max_iou] = torch.LongTensor(range(n_objs)).to(device)

        # Fix situation #2.
        max_iou_for_each_prior[prior_at_max_iou] = 1.0

    def encode_gt_bbox(self, box, obj_at_max_iou):
        return cxcy_to_gcxgcy(xy_to_cxcy(box[obj_at_max_iou]), self.priors_cxcy)

    def calc_loc_loss(self, pred_locs, true_locs, pos_priors):
        return self.smooth_l1(pred_locs[pos_priors], true_locs[pos_priors])

    def calc_confidence_loss(
        self, pred_scores, true_clss, pos_priors, ncls, npriors, bs
    ):
        """
                Calculate confidence loss for each object in positive priors.

                Confidence loss is computed over positive priors
                and the most difficult (hardest) negative priors in each image.
        reture
                This is called Hard Negative Mining - it concentrates on hardest negatives in each image,
                and also minimizes pos/neg imbalance.

                Args:
                    pred_scores: the prediction scores made by our model.

                    true_clss: the ground truth labels for each object.

                    pos_priors: priors that contain objects (non-background).

                    ncls: number of classes in dataset (20 for VOC + background = 21).

                    npriors: number of priors returned by model. (8732 according to paper)

                    bs: the batch size.

        """

        n_pos = pos_priors.sum(dim=1)
        n_hard_negs = self.neg_pos_ratio * n_pos

        conf_loss_all = self.cross_entropy(
            pred_scores.view(-1, ncls), true_clss.view(-1)
        )
        conf_loss_all = conf_loss_all.view(bs, npriors)
        assert conf_loss_all.shape == (bs, 8732)

        conf_loss_pos = conf_loss_all[pos_priors]

        conf_loss_hard_neg = self.hard_negative_mining(
            conf_loss_all, pos_priors, n_pos, n_hard_negs, npriors
        )

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (
            conf_loss_hard_neg.sum() + conf_loss_pos.sum()
        ) / n_pos.sum().float()

        return conf_loss

    def hard_negative_mining(
        self, conf_loss_all, pos_priors, n_pos, n_hard_negs, npriors
    ):
        # sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[
            pos_priors
        ] = 0.0  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)

        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732)

        hardness_ranks = (
            torch.LongTensor(range(npriors))
            .unsqueeze(0)
            .expand_as(conf_loss_neg)
            .to(device)
        )  # (N, 8732)

        hard_negs = hardness_ranks < n_hard_negs.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negs]

        return conf_loss_hard_neg
