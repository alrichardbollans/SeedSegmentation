import os
import shutil

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from ultralytics.data.converter import convert_coco

_repo_path = os.environ['KEWSCRATCHPATH']
REPO_PATH = os.path.join(_repo_path, 'SeedSegmentation')

from pycocotools.coco import COCO

basic_annotation_label = 'Seed'
detailed_labels = ['Viable', 'Non viable']


def load_coco_annotations(json_file: str):
    coco = COCO(json_file)

    # Get category IDs and annotation IDs
    catIds = coco.getCatIds()
    annsIds = coco.getAnnIds()

    # for ann in annsIds:
    #     # Get individual masks
    #     loaded = coco.loadAnns(ann)
    #     mask = coco.annToMask(loaded[0])
    #     print(mask)
    #
    # with open(json_file) as f:
    #     d = json.load(f)
    #     print(d)

    return coco


def draw_coco_annotation(coco, img_id: int, img_dir: str, outpath: str, annotator: str = None, f1: float = None):
    if annotator is None:
        annotator = ''
    from PIL import Image

    img_info = coco.loadImgs(img_id)[0]

    img = Image.open(os.path.join(img_dir, img_info['file_name']))

    # Get annotations for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    # Plot the image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.axis('off')

    # Create a list to store polygons and a list for their colors
    polygons = []
    colors = []
    labels = []
    count = 0
    for ann in anns:
        category_ids = coco.getCatIds(catIds=ann['category_id'])
        categories = coco.loadCats(category_ids)
        category_name = categories[0]['name']
        if len(categories) > 1:
            raise ValueError('Unexpected, check this')
        if 'segmentation' in ann:
            if category_name == basic_annotation_label or category_name in detailed_labels:
                count += 1
            else:
                raise ValueError(f'Unexpected label: {category_name}')
            if len(ann['segmentation']) > 1:
                raise ValueError('Unexpected, check this')
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                # plt.plot(poly[:, 0], poly[:, 1], linewidth=2)
                polygons.append(Polygon(poly))

                # Generate a random color for each polygon
                colors.append(np.concatenate([np.random.random(3), [0.35]]))

                labels.append(category_name)
    # Create a PatchCollection of polygons
    p = PatchCollection(polygons, facecolors=colors, edgecolors=(0, 0, 0, 1), linewidths=2, alpha=0.4)
    ax.add_collection(p)

    # Annotate polygons with category names
    for poly, label in zip(polygons, labels):
        # Get the center of the polygon for text placement
        x, y = poly.get_xy().mean(axis=0)
        ax.annotate(label, (x, y), color='black', weight='bold', fontsize=8, ha='center', va='center')

    title = f'{img_info["file_name"]}. Seed count: {count}. Annotator: {annotator}'
    if f1 is not None:
        title += f'. F1 score: {round(f1, 2)}'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, f'{img_info["file_name"]}_{annotator}.png'))


def validate_annotation(annotation):
    # Check label is known
    # Check image size is same as original
    # Check is mask or polygon, depending on what decide to use

    # Check each has single category

    # Check each has single segmentation
    pass


def convert_coco_labels_to_basic(coco):
    '''
    Convert viable/nonviable annotations to 'seed' annotations
    :param coco:
    :return:
    '''
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        ann['category_id'] = 1

    setattr(coco, 'cats', {1: {'id': 1, 'name': basic_annotation_label}})
    return coco


def make_yaml_file_for_coco_data(train_images_path, train_annotations_path, val_images_path, val_annotations_path, outfile: str):
    # https://github.com/ultralytics/ultralytics/issues/9161
    coco = load_coco_annotations(train_annotations_path)
    classes = coco.loadCats(coco.getCatIds())
    stri = f'train: "{train_images_path}"\nval: "{val_images_path}"\ntrain_ann: "{train_annotations_path}"\nval_ann: "{val_annotations_path}"\nnc: {len(classes)}\nnames: {[x["name"] for x in classes]}'
    with open(outfile, 'w') as f:
        f.write(stri)


def convert_coco_to_yolo(img_dir, labels_dir,
                         save_dir,
                         use_segments=True,
                         use_keypoints=False,
                         cls91to80=True,
                         lvis=False):
    # This is just a wrapper for the ultralytics function to remember it exists
    # Also moves images over to images/default
    # https://docs.ultralytics.com/reference/data/converter/#ultralytics.data.converter.convert_coco
    # Note it seems to change category ids <- -1
    if os.path.exists(save_dir):
        raise ValueError(f"the convert_coco function will make new dirs which I dont want. If this raises error then delete folder: {save_dir}")
    convert_coco(labels_dir=labels_dir,
                 save_dir=save_dir,
                 use_segments=use_segments,
                 use_keypoints=use_keypoints,
                 cls91to80=cls91to80,
                 lvis=lvis)
    # Organise directories following: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#22-create-labels
    # i.e. labels located by replacing the last instance of /images/ in each image path with /labels/
    os.makedirs(os.path.join(save_dir, 'images', 'default'), exist_ok=True)
    files = os.listdir(img_dir)
    for fname in files:
        shutil.copy2(os.path.join(img_dir, fname), os.path.join(save_dir, 'images', 'default'))


def make_yaml_file_for_yolo_data(train_images_path, val_images_path, classes, outfile: str):
    # https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#22-create-labels

    stri = f'train: "{train_images_path}"\nval: "{val_images_path}"\nnc: {len(classes)}\nnames: {classes}'
    with open(outfile, 'w') as f:
        f.write(stri)


if __name__ == '__main__':
    human_coco = load_coco_annotations(
        os.path.join(REPO_PATH, 'data', 'annotations', 'pablo_examples', 'coco_polygons', 'annotations',
                     'instances_default.json'))
    updated_human_coco = convert_coco_labels_to_basic(human_coco)

    model_coco = load_coco_annotations(
        os.path.join(REPO_PATH, 'two step models', 'seed segmentation', 'example_outputs', 'coco_outputs', 'result.json'))
