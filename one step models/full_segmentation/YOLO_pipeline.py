import ast
import json
import os

import cv2
from ultralytics import YOLO

from myutils.annotation_processing import REPO_PATH, convert_coco_to_yolo, make_yaml_file_for_yolo_data


def yolo_instance(model, file_image: str, outdir: str):
    img = cv2.imread(file_image)

    conf = 0.01  # Confidence threshold

    results = model.predict(img, conf=conf)
    assert len(results) == 1  # Should only have a single result i.e. 1 image. If not, then need to change this
    classes = results[0].boxes.cls  # Probs object for classification outputs

    # Count detected objects
    print(f'Number of objects detected: {len(results[0].masks)}')
    object_counts = {}
    for obj in classes:
        name = results[0].names[int(obj)]
        if name in object_counts:
            object_counts[name] = object_counts[name] + 1
        else:
            object_counts[name] = 1
    print(f'objects detected: {object_counts}')
    annotated_image = results[0].plot(boxes=True)
    try:
        number_seeds = object_counts['Seed']
    except KeyError:
        number_seeds = 0

    # Plot
    os.makedirs(os.path.join(outdir, 'annotated_images'), exist_ok=True)
    results[0].save(os.path.join(outdir, 'annotated_images', f"{os.path.basename(file_image)}_YOLO.png"))

    return number_seeds, ast.literal_eval(results[0].tojson()), img.shape


def yolo_training_example():
    _data_path = os.environ['KEWDATAPATH']
    yolo_seg_pt = os.path.join(_data_path, 'model_checkpoints', 'YOLO', "yolov8n-seg.pt")
    model = YOLO(yolo_seg_pt)  # load a pretrained model (recommended for training)

    # Train the model
    # TODO: Add 'background# training images as in https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/

    # Need to convert the coco data into YOLO format. In theory ultralytics can handle the coco format but
    # it wasn't working https://github.com/ultralytics/ultralytics/issues/9161#issuecomment-2275879820
    # alternatively, CVAT yolo output doesn't seem to work either, so this is a work around

    # Will need to play around with this to also include testing examples etc..
    convert_coco_to_yolo(os.path.join(REPO_PATH, 'data', 'annotations', 'pablo_examples', 'images'),
                         os.path.join(REPO_PATH, 'data', 'annotations', 'pablo_examples', 'coco_polygons', 'annotations'),
                         os.path.join(REPO_PATH, 'data', 'annotations', 'pablo_examples', 'yolo_format'))
    yaml_file = os.path.join(REPO_PATH, 'data', 'annotations', 'pablo_examples', 'yolo_format', 'dataset.yaml')
    make_yaml_file_for_yolo_data(os.path.join(REPO_PATH, 'data', 'annotations', 'pablo_examples', 'yolo_format', 'images', 'default'),
                                 os.path.join(REPO_PATH, 'data', 'annotations', 'pablo_examples', 'yolo_format', 'images', 'default'),
                                 ['Viable', 'Non viable'], yaml_file)

    # For some reason, seems to be downloading a new model
    results = model.train(
        data=yaml_file, single_cls=False,
        epochs=100)


def load_trained_model(train_no, runs_dir: str = '/home/atp/runs'):
    model = YOLO(os.path.join(runs_dir, 'segment', 'train' + str(train_no), 'weights', 'best.pt'))
    yolo_instance(model, os.path.join(REPO_PATH, 'data', 'annotations', 'pablo_examples', 'images',
                                      '394660_00.jpg'), os.path.join('example_outputs', 'trained_yolo_output'))


def yolo_Evaluation_example():
    ### Try evaluating from annotations
    pass


def example_of_saving_annotations():
    # TODO: Not currently useful
    # TODO: will need to play with categories etc here and how results are saved..
    # TODO: Need to fix so that works with load_coco_annotations, or set up separate evaluation for this
    # Load a model
    _data_path = os.environ['KEWDATAPATH']
    # See https://docs.ultralytics.com/tasks/segment/#models for different models
    yolo_seg_pt = os.path.join(_data_path, 'model_checkpoints', 'YOLO', "yolov8n-seg.pt")
    model = YOLO(yolo_seg_pt)  # load a pretrained model (recommended for training)

    out_dict = {'licences': [{'id': 0, 'name': '', 'url': ''}],
                'info': {'contributor': 'YOLO', 'date_created': '', 'description': '', 'url': '', 'version': '', 'year': ''},
                'categories': [{'id': 1, 'name': 'Seed'}], 'images': [], 'annotations': []}
    ann_id = 1
    img_id = 1
    for file_image in os.listdir(os.path.join(REPO_PATH, 'data', 'annotations', 'pablo_examples', 'images')):
        num_seeds, results, image_shape = yolo_instance(model,
                                                        os.path.join(REPO_PATH, 'data', 'annotations', 'pablo_examples', 'images',
                                                                     file_image), 'example_outputs')
        out_dict['images'].append({'file_name': file_image, 'height': image_shape[0], 'width': image_shape[1], 'id': img_id})
        for r in results:
            r['image_id'] = img_id
            r['id'] = ann_id
            ann_id += 1
        out_dict['annotations'].append(results)
        img_id += 1
        with open(os.path.join('example_outputs', 'coco_outputs', 'yolo_results.json'), 'w') as fp:
            json.dump(out_dict, fp)


if __name__ == '__main__':
    # yolo_training_example()
    load_trained_model(45)
