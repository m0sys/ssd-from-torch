import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import torch


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """ wrapper function for endless data loader. """
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


# ------------------------------- SSD Utils ------------------------------------------------- #


def decimate(tensor, m):
    """
    Decimate a tensor by a factor of 'm', i.e. downsample by keeping every 'm'th value.
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(
                dim=d, index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long()
            )
    return tensor


def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coords (x_min, y_min, x_max, y_max) to
    center-size coords (c_x, c_y, w, h).

    Args:
        xy: bounding boxes in boundary coords - a tensor of size (n_boxes, 4)

    Returns:
        bounding boxes in center-size coords - a tensor of size(n_boxes, 4)
    """
    return torch.cat(
        [(xy[:, 2:] + xy[:, :2]) / 2, (xy[:, 2:] - xy[:, :2])], 1  # (c_x, c_y), (w, h)
    )


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coords (c_x, c_y, w, h) to boundary coords
    (x_min, y_min, x_max, y_max).

    Args:
        cxcy: bounding boxes in center-size coords - a tensor of size (n_boxes, 4)

    Returns:
        xy: bounding boxes in boundary coords - a tensor of size (n_boxes, 4)

    """

    return torch.cat(
        [cxcy[:, :2] - (cxcy[:, 2:] / 2), cxcy[:, :2] + (cxcy[:, 2:] / 2)],
        1,  # (x_min, y_min), (x_max, y_max)
    )


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes w.r.t. the corresponding prior boxes.

    For the center coordiantes, find the offset w.r.t. the prior box, and
    scale by the size of the prior box.

    For the size coordinates, scale by the size of the prior box, and
    covert to the log-space.
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155

    return torch.cat(
        [
            (cxcy[:, :2] - priors_cxcy[:, :2])
            / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
            torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5,
        ],
        1,
    )  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.
    """

    return torch.cat(
        [
            gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
            torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:],
        ],
        1,
    )  # w, h


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of
    boxes that are in boundary coordinates.


    Args:
        set_1: a tensor of dimensions (n1, 4)
        set_2: a tensor of dimensions (n2, 4)

    Returns:
        a tensor of dimensions (n1, n2)
    """

    lower_bounds = torch.max(
        set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0)
    )  # (n1, n2, 2)
    upper_bounds = torch.min(
        set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0)
    )  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the IOU (intersection over union) of every combination between two
    sets of boxes that are in boundary containers.


    Args:
        set_1: a tensor of dimensions (n1, 4)
        set_2: a tensor of dimensions (n2, 4)

    Returns:
        a tensor of dimensions (n1, n2)
    """
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    areas_set_1 = _find_area(set_1)  # (n1)
    areas_set_2 = _find_area(set_2)  # (n2)

    union = (
        areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection
    )  # (n1, n2)

    return intersection / union  # (n1, n2)


def _find_area(sset):
    """
    Find area of boxes that are in boundary container.

    Args:
        sset: a tensor of dimensions (n, 4)

    Returns:
        a tensor of dimensions (n)
    """

    return (sset[:, 2] - sset[:, 0]) * (sset[:, 3] - sset[:, 1])


def adjust_lr(optimizer, scale):
    """
    Scale lr by a specified factor.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] *= scale

    print(
        "DECAYING learning rate.\n The new LR is %f\n"
        % (optimizer.param_groups[1]["lr"],)
    )


def save_checkpoint(epoch, model, optimizer):
    state = {"epoch": epoch, "model": model, "optimizer": optimizer}

    fname = "checkpoint_ssd300.pth.tar"
    torch.save(state, fname)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    Args:
        optimizer: optimizer with the gradients to be clipped
        grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
