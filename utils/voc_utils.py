import json
import os
import xml.etree.ElementTree as ET

voc_labels = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map["background"] = 0
rev_label_map = {v: k for k, v in label_map.items()}


# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#000080",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#e6beff",
    "#808080",
    "#FFFFFF",
]


label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


def create_data_lists(voc07_path, voc12_path, output_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images,
    and save these to file.
    :param voc07_path: path to the 'VOC2007' folder
    :param voc12_path: path to the 'VOC2012' folder
    :param output_folder: folder where the JSONs must be saved
    """
    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    _voc2coco(paths=[voc07_path, voc12_path], outdir=output_folder)
    _voc2coco(paths=[voc07_path], outdir=output_folder, train=False)

    _save_to_json(output_folder, "label_map.json", label_map)


def _voc2coco(paths, outdir, train=True):
    images = []
    objects = []
    n_objects = 0

    # Train data.
    for path in paths:
        ids = _get_voc_img_ids(
            os.path.join(
                path,
                "ImageSets/Main/trainval.txt" if train else "ImageSets/Main/val.txt",
            )
        )

        for iD in ids:
            anno_objs = parse_annotation(os.path.join(path, "Annotations", iD + ".xml"))

            if train:

                if len(anno_objs["boxes"]) == 0:
                    continue
            else:
                if len(anno_objs) == 0:
                    continue

            n_objects += len(anno_objs)
            objects.append(anno_objs)

            images.append(os.path.join(path, "JPEGImages", iD + ".jpg"))

    assert len(objects) == len(images)

    _save_to_json(outdir, f"{'TRAIN' if train else 'TEST'}_images.json", images)
    _save_to_json(outdir, f"{'TRAIN' if train else 'TEST'}_objects.json", objects)

    print(
        f"\nThere are {len(images)} {'training' if train else 'test'} images containing a totla of {n_objects} objects. Files have been saved to {os.path.abspath(outdir)}"
    )


def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter("object"):

        difficult = int(object.find("difficult").text == "1")

        label = object.find("name").text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find("bndbox")
        xmin = int(bbox.find("xmin").text) - 1
        ymin = int(bbox.find("ymin").text) - 1
        xmax = int(bbox.find("xmax").text) - 1
        ymax = int(bbox.find("ymax").text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {"boxes": boxes, "labels": labels, "difficulties": difficulties}


def _get_voc_img_ids(path):
    with open(path) as f:
        ids = f.read().splitlines()
    return ids


def _save_to_json(outdir, filename, obj):
    with open(os.path.join(outdir, filename), "w") as j:
        json.dump(obj, j)