import random
import torch
import torchvision.transforms.functional as FT

from utils.util import find_jaccard_overlap


def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filter material.

    This helps to detect smaller objects.

    Args:
        image: an image tensor of dim (3, original_h, original_w)
        boxes: bounding boxes in boundary coordinates - a tensor of dim (n_objs, 4)
        filler: RGB values of the filter material - list like [R, G, B]

    Returns:
        expanded image and updated bouding box coords that match the transform.
    """

    org_h = image.size(1)
    org_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * org_h)
    new_w = int(scale * org_w)

    # Create image filler (canvas).
    filler = torch.FloatTensor(filler)
    new_img = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(
        1
    ).unsqueeze(1)

    # Place original image at random coords in canvas.
    left = random.randint(0, new_w - org_w)
    right = left + org_w
    top = random.randint(0, new_h - org_h)
    bot = top + org_h
    new_img[:, top:bot, left:right] = image

    # Adjust bounding box coords accordingly.
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)

    return new_img, new_boxes


def random_crop(image, boxes, labels, difficulties):
    """
    Perform a random crop in the manner stated in paper.

    This helps to detect larger objects and partial objs.

    Note that some objs might be cut out entirely.

    Args:
        image: an image tensor of dim (3, original_h, original_w)
        boxes: bounding boxes in boundary coordinates - a tensor of dim (n_objs, 4)
        filler: RGB values of the filter material - list like [R, G, B]

    Returns:
        cropped image and updated bouding box coords, labels, and difficulties
        that match the transform.
    """
    org_h = image.size(1)
    org_w = image.size(2)

    # Keep choosing a min overlap until a succesful crop is made.
    while True:
        min_overlap = random.choice([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, None])

        if min_overlap is None:
            return image, boxes, labels, difficulties

        max_trials = 50
        while max_trials:
            max_trials -= 1

            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * org_h)
            new_w = int(scale_w * org_w)

            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            left = random.randint(0, org_w - new_w)
            right = left + new_w
            top = random.randint(0, org_h - new_h)
            bot = top + new_h
            crop = torch.FloatTensor([left, top, right, bot])

            overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes)

            overlap = overlap.squeeze(0)

            # If not a single bbox has a IOU overlap of greater than min, try again.
            if overlap.max().item() < min_overlap:
                continue

            new_img = image[:, top:bot, left:right]

            # Find og boxes whose centers are in the crop.
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

            # Find bboxes whose centers are in the crop.
            centers_in_crop = (
                (bb_centers[:, 0] > left)
                * (bb_centers[:, 0] < right)
                * (bb_centers[:, 1] > top)
                * (bb_centers[:, 1] < bot)
            )

            # If not a signle bbox has its center in the crop, try again.
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(
                new_boxes[:, :2], crop[:2]
            )  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(
                new_boxes[:, 2:], crop[2:]
            )  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_img, new_boxes, new_labels, new_difficulties


def flip(image, boxes):
    """
    Flip image horizontally.
    Args:
        image: image, a PIL Image
        boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)

    Returns:
        flipped image, updated bounding box coordinates
    """
    # Flip image
    new_img = FT.hflip(image)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_img, new_boxes


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).

    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.

    Args:
        image: image, a PIL Image
        boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)


    Returns:
        resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor(
        [image.width, image.height, image.width, image.height]
    ).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.

    Args:
        image: image, a PIL Image
    Returns:
        distorted image
    """
    new_img = image

    distortions = [
        FT.adjust_brightness,
        FT.adjust_contrast,
        FT.adjust_saturation,
        FT.adjust_hue,
    ]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ == "adjust_hue":
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255.0, 18 / 255.0)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_img = d(new_img, adjust_factor)

    return new_img


def transform(image, boxes, labels, difficulties, split):
    """
    Apply all transformations in paper.

    Args:
        image: image, a PIL Image
        boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
        labels: labels of objects, a tensor of dimensions (n_objects)
        difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
        split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied

    Returns:
        transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """
    assert split in {"TRAIN", "TEST"}

    # Mean and standard deviation of ImageNet data for base VGG model.
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_img = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties

    if split == "TRAIN":
        new_img = photometric_distort(new_img)

        new_img = FT.to_tensor(new_img)
        if random.random() < 0.5:
            new_img, new_boxes = expand(new_img, boxes, filler=mean)

        new_img, new_boxes, new_labels, new_difficulties = random_crop(
            new_img, new_boxes, new_labels, new_difficulties
        )

        new_img = FT.to_pil_image(new_img)
        if random.random() < 0.5:
            new_img, new_boxes = flip(new_img, new_boxes)

    new_img, new_boxes = resize(new_img, new_boxes, dims=(300, 300))
    new_img = FT.to_tensor(new_img)
    new_img = FT.normalize(new_img, mean=mean, std=std)

    return new_img, new_boxes, new_labels, new_difficulties