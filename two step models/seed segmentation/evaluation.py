import os
from collections import defaultdict

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from pycocotools.cocoeval import COCOeval
from matplotlib import pyplot as plt

from myutils.annotation_processing import load_coco_annotations, draw_coco_annotation, REPO_PATH, convert_coco_labels_to_basic


def calculate_iou(poly1, poly2, draw: bool = False):
    # TODO: Check this

    shape1 = Polygon(np.array(poly1).reshape(-1, 2))
    shape2 = Polygon(np.array(poly2).reshape(-1, 2))
    intersection = shape1.intersection(shape2).area
    union = shape1.union(shape2).area
    iou = intersection / union if union > 0 else 0
    if draw:
        plt.plot(*shape1.exterior.xy)
        plt.plot(*shape2.exterior.xy)
        plt.title(f'iou:{round(iou, 2)}')
        plt.show()

    return iou


def calculate_average_precision(iou_threshold=0.5):
    # TODO: implement

    pass


def calculate_mean_average_precision():
    # TODO: implement

    pass


def compare_number_of_seeds(human_annotations, model_annotations):
    # TODO: implement

    pass


def get_img_info_from_coco_annotations(coco_annotations):
    _img_ids = coco_annotations.getImgIds()
    _img_info = coco_annotations.loadImgs(_img_ids)
    return _img_info


def evaluate_annotations(human_annotations, model_annotations, img_file_name: str, iou_threshold=0.5):
    # TODO: refactor
    # TODO: Add other metrics
    # TODO: Check this

    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)

    human_ann_ids = human_annotations.getAnnIds()
    human_anns = human_annotations.loadAnns(human_ann_ids)
    human_img_info = get_img_info_from_coco_annotations(human_annotations)
    relevant_human_img_info = [x for x in human_img_info if x['file_name'] == img_file_name]
    assert len(relevant_human_img_info) == 1
    human_img_id = relevant_human_img_info[0]['id']

    model_ann_ids = model_annotations.getAnnIds()
    model_anns = model_annotations.loadAnns(model_ann_ids)
    model_img_info = get_img_info_from_coco_annotations(model_annotations)
    relevant_model_img_info = [x for x in model_img_info if x['file_name'] == img_file_name]
    assert len(relevant_model_img_info) == 1
    model_img_id = relevant_model_img_info[0]['id']

    for gt_ann in human_anns:
        if gt_ann['image_id'] == human_img_id:
            gt_cat = gt_ann['category_id']
            gt_poly = gt_ann['segmentation'][0]  # Assuming single polygon per annotation
            assert len(gt_ann['segmentation']) == 1

            best_iou = 0
            best_match = None

            for pred_ann in model_anns:
                if pred_ann['image_id'] == model_img_id:

                    pred_cat = pred_ann['category_id']
                    pred_poly = pred_ann['segmentation'][0]  # Assuming single polygon per annotation
                    assert len(pred_ann['segmentation']) == 1
                    if gt_cat == pred_cat:
                        iou = calculate_iou(gt_poly, pred_poly)
                        if iou > best_iou:
                            best_iou = iou
                            best_match = pred_ann

            if best_iou >= iou_threshold:
                true_positives[gt_cat] += 1
                model_anns.remove(best_match)
            else:
                false_negatives[gt_cat] += 1

    # Remaining predictions are false positives
    for pred_ann in model_anns:
        if pred_ann['image_id'] == model_img_id:
            false_positives[pred_ann['category_id']] += 1

    # Calculate metrics
    categories = set(ann['category_id'] for ann in human_anns + model_anns)
    results = {}

    # for cat in categories:
    #     tp = true_positives[cat]
    #     fp = false_positives[cat]
    #     fn = false_negatives[cat]
    #
    #     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    #     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    #     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    #
    #     results[cat] = {
    #         'precision': precision,
    #         'recall': recall,
    #         'f1_score': f1
    #     }

    # Calculate overall metrics
    total_tp = sum(true_positives.values())
    total_fp = sum(false_positives.values())
    total_fn = sum(false_negatives.values())

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    results[img_file_name] = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1
    }
    out_df = pd.DataFrame.from_dict(results)
    return out_df


def SAM_example():
    human_coco = load_coco_annotations(
        os.path.join(REPO_PATH, 'orchid_tz', 'data', 'annotations', 'pablo examples', 'coco_polygons', 'annotations', 'instances_default.json'))
    model_coco = load_coco_annotations(
        os.path.join(REPO_PATH, 'orchid_tz', 'two step models', 'seed segmentation', 'example_outputs', 'coco_outputs', 'result.json'))
    updated_human_coco = convert_coco_labels_to_basic(human_coco)
    all_img_info = get_img_info_from_coco_annotations(model_coco)

    evaluated = pd.DataFrame()
    for img in all_img_info:
        img_filename = img['file_name']
        df = evaluate_annotations(updated_human_coco, model_coco, img_filename)
        evaluated = pd.concat([evaluated, df], axis=1)

    evaluated['overall'] = evaluated.mean(axis=1)
    evaluated.to_csv('50.csv')

    # TODO: make a note that coco IDs from different sets of annotations may not resolve to same image file. Maybe fix this
    ids = updated_human_coco.getImgIds()
    for i in ids:
        draw_coco_annotation(updated_human_coco, ids[i - 1], os.path.join(REPO_PATH, 'orchid_tz', 'data', 'annotations', 'pablo examples', 'images'),
                             os.path.join(REPO_PATH, 'orchid_tz', 'two step models', 'seed segmentation', 'example_outputs'),
                             annotator='Pablo')

    ids = model_coco.getImgIds()
    for i in ids:
        img_info = model_coco.loadImgs(i)[0]
        img_filename = img_info['file_name']
        f1 = evaluated[img_filename]['f1_score']
        draw_coco_annotation(model_coco, ids[i - 1], os.path.join(REPO_PATH, 'orchid_tz', 'data', 'annotations', 'pablo examples', 'images'),
                             os.path.join(REPO_PATH, 'orchid_tz', 'two step models', 'seed segmentation', 'example_outputs'), f1=f1,
                             annotator='SAM')


def YOLO_example():
    human_coco = load_coco_annotations(
        os.path.join(REPO_PATH, 'orchid_tz', 'data', 'annotations', 'pablo examples', 'coco_polygons', 'annotations', 'instances_default.json'))
    model_coco = load_coco_annotations(
        os.path.join(REPO_PATH, 'orchid_tz', 'two step models', 'seed segmentation', 'example_outputs', 'coco_outputs', 'yolo_results.json'))
    updated_human_coco = convert_coco_labels_to_basic(human_coco)
    all_img_info = get_img_info_from_coco_annotations(model_coco)



if __name__ == '__main__':
    # SAM_example()
    YOLO_example()
